"""FastAPI inference service for pneumonia X-ray classification."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.core.config import ConfigFileError, load_config
from src.data.dataset import SUPPORTED_IMAGE_EXTENSIONS
from src.inference.predict import (
    InferenceError,
    LoadedModel,
    load_model,
    predict_image,
    preprocess_image,
)

MEDICAL_WARNING = "This output must not be used for medical diagnosis."
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
MODEL_PATH_ENVIRONMENT_VARIABLE = "PNEUMONIA_MODEL_PATH"
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024


class HealthResponse(BaseModel):
    """Health endpoint response schema.

    Attributes:
        status: Service status value.
        model_loaded: Whether the model is currently loaded in application state.
    """

    status: str = Field(examples=["ok"])
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Model metadata response schema.

    Attributes:
        model_name: Model architecture name from the checkpoint.
        model_version: Checkpoint filename used by the API.
        checkpoint_path: Checkpoint path configured for the service.
        image_size: Input image size expected by the model.
        label_mapping: Class-label-to-id mapping used by the checkpoint.
        device: Runtime device where the model is loaded.
        warning: Medical-use warning shown with all model-facing responses.
    """

    model_name: str
    model_version: str
    checkpoint_path: str
    image_size: int
    label_mapping: dict[str, int]
    device: str
    decision_threshold: float
    preprocessing: str
    warning: str


class PredictionResponse(BaseModel):
    """Prediction endpoint response schema.

    Attributes:
        predicted_label: Predicted class label.
        normal_probability: Probability assigned to the NORMAL class.
        pneumonia_probability: Probability assigned to the PNEUMONIA class.
        confidence: Probability assigned to the predicted class.
        warning: Medical-use warning.
    """

    predicted_label: str
    normal_probability: float = Field(ge=0.0, le=1.0)
    pneumonia_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    warning: str


class ErrorResponse(BaseModel):
    """User-facing API error response schema.

    Attributes:
        detail: Human-readable error message.
    """

    detail: str


def create_app(
    *,
    checkpoint_path: str | Path | None = None,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> FastAPI:
    """Create a FastAPI app that loads the model once during startup.

    Args:
        checkpoint_path: Optional explicit checkpoint path. If omitted, the app uses
            `PNEUMONIA_MODEL_PATH`, then `configs/config.yaml`.
        config_path: Project configuration path used to locate the default checkpoint.

    Returns:
        Configured FastAPI application.
    """

    resolved_checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
    )

    @asynccontextmanager
    async def lifespan(api_app: FastAPI) -> AsyncIterator[None]:
        try:
            api_app.state.loaded_model = load_model(resolved_checkpoint_path)
        except (FileNotFoundError, InferenceError) as error:
            message = f"Could not initialize the API model: {error}"
            raise RuntimeError(message) from error
        yield

    api_app = FastAPI(
        title="Pneumonia X-Ray CNN API",
        description="An API that predicts NORMAL/PNEUMONIA classes from chest X-rays.",
        version="0.1.0",
        lifespan=lifespan,
    )
    api_app.state.checkpoint_path = resolved_checkpoint_path

    @api_app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: object,
        exception: RequestValidationError,
    ) -> JSONResponse:
        """Convert missing upload validation errors to a user-facing response.

        Args:
            _request: Request object supplied by FastAPI.
            exception: Validation exception raised before endpoint execution.

        Returns:
            JSON error response.
        """

        missing_file_errors = [
            error
            for error in exception.errors()
            if tuple(error.get("loc", ())) == ("body", "file") and error.get("type") == "missing"
        ]
        if missing_file_errors:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "detail": "No image file was provided; the multipart `file` field is required."
                },
            )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "Request validation failed; check the multipart image upload."},
        )

    @api_app.get(
        "/health",
        response_model=HealthResponse,
        responses={status.HTTP_200_OK: {"model": HealthResponse}},
    )
    def health() -> HealthResponse:
        """Return service health and model-load status.

        Returns:
            Health status response.
        """

        return HealthResponse(
            status="ok",
            model_loaded=isinstance(getattr(api_app.state, "loaded_model", None), LoadedModel),
        )

    @api_app.get(
        "/model-info",
        response_model=ModelInfoResponse,
        responses={
            status.HTTP_200_OK: {"model": ModelInfoResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
        },
    )
    def model_info() -> ModelInfoResponse:
        """Return metadata for the loaded checkpoint.

        Returns:
            Model metadata response.

        Raises:
            HTTPException: If the model is unavailable.
        """

        loaded_model = get_loaded_model(api_app)
        return ModelInfoResponse(
            model_name=loaded_model.model_name,
            model_version=loaded_model.model_version or loaded_model.checkpoint_path.name,
            checkpoint_path=str(loaded_model.checkpoint_path),
            image_size=loaded_model.image_size,
            label_mapping=loaded_model.label_mapping,
            device=str(loaded_model.device),
            decision_threshold=loaded_model.decision_threshold,
            preprocessing=loaded_model.preprocessing,
            warning=MEDICAL_WARNING,
        )

    @api_app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            status.HTTP_200_OK: {"model": PredictionResponse},
            status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"model": ErrorResponse},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse},
        },
    )
    async def predict(file: Annotated[UploadFile, File(...)]) -> PredictionResponse:
        """Predict pneumonia probability from a multipart image upload.

        Args:
            file: Uploaded chest X-ray image file.

        Returns:
            Prediction response with class probabilities and medical-use warning.

        Raises:
            HTTPException: If validation, preprocessing, or inference fails.
        """

        loaded_model = get_loaded_model(api_app)
        validate_upload_metadata(file)
        image_bytes = await read_upload_bytes(file)
        try:
            image_tensor = preprocess_image(
                image_bytes,
                image_size=loaded_model.image_size,
                normalization_mean=loaded_model.normalization_mean,
                normalization_std=loaded_model.normalization_std,
                preprocessing=loaded_model.preprocessing,
            )
            prediction = predict_image(loaded_model, image_tensor)
        except InferenceError as error:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not process the image: {error}",
            ) from error

        return PredictionResponse(
            predicted_label=str(prediction["predicted_label"]),
            normal_probability=float(prediction["normal_probability"]),
            pneumonia_probability=float(prediction["pneumonia_probability"]),
            confidence=float(prediction["confidence"]),
            warning=MEDICAL_WARNING,
        )

    return api_app


def resolve_checkpoint_path(
    *,
    checkpoint_path: str | Path | None,
    config_path: str | Path,
) -> Path:
    """Resolve the model checkpoint path used by the API.

    Args:
        checkpoint_path: Explicit checkpoint override.
        config_path: Configuration path used when no override is supplied.

    Returns:
        Resolved checkpoint path.

    Raises:
        RuntimeError: If the configuration cannot be loaded.
    """

    if checkpoint_path is not None:
        return Path(checkpoint_path)

    environment_checkpoint_path = os.getenv(MODEL_PATH_ENVIRONMENT_VARIABLE)
    if environment_checkpoint_path:
        return Path(environment_checkpoint_path)

    try:
        config = load_config(config_path)
    except ConfigFileError as error:
        message = f"Could not resolve the API checkpoint path; configuration read failed: {error}"
        raise RuntimeError(message) from error
    return config.paths.best_model_path


def get_loaded_model(api_app: FastAPI) -> LoadedModel:
    """Fetch the startup-loaded model from FastAPI state.

    Args:
        api_app: FastAPI application instance.

    Returns:
        Loaded model bundle.

    Raises:
        HTTPException: If no loaded model exists in application state.
    """

    loaded_model = getattr(api_app.state, "loaded_model", None)
    if not isinstance(loaded_model, LoadedModel):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model has not loaded yet; check the service startup status.",
        )
    return loaded_model


def validate_upload_metadata(file: UploadFile) -> None:
    """Validate multipart upload metadata before reading the file body.

    Args:
        file: Uploaded file object.

    Raises:
        HTTPException: If filename or content type is unsupported.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file was provided; the multipart `file` field is required.",
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        supported_extensions = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image format: {suffix or 'no extension'}. "
                f"Supported extensions: {supported_extensions}"
            ),
        )

    if file.content_type is not None and not file.content_type.lower().startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type: {file.content_type}. "
                "Upload a JPEG, JPG, or PNG image."
            ),
        )


async def read_upload_bytes(file: UploadFile) -> bytes:
    """Read and size-check an uploaded image.

    Args:
        file: Uploaded file object.

    Returns:
        Uploaded file bytes.

    Raises:
        HTTPException: If the file is empty, too large, or unreadable.
    """

    try:
        image_bytes = await file.read()
    except OSError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read the uploaded file: {error}",
        ) from error

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty.",
        )
    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:
        maximum_megabytes = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"The uploaded image exceeds the {maximum_megabytes} MB limit.",
        )
    return image_bytes


app = create_app()
