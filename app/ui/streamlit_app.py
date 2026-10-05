"""Streamlit demo UI for pneumonia X-ray prediction through the FastAPI service."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException, HTTPSConnection
from pathlib import Path
from typing import Final, Protocol
from urllib.parse import ParseResult, urlparse

import streamlit as st

APP_TITLE: Final[str] = "Pneumonia Classification from Chest X-Rays"
MEDICAL_WARNING: Final[str] = "This output must not be used for medical diagnosis."
DEFAULT_API_BASE_URL: Final[str] = "http://127.0.0.1:8000"
API_BASE_URL_ENVIRONMENT_VARIABLE: Final[str] = "PNEUMONIA_API_URL"
SUPPORTED_UPLOAD_TYPES: Final[tuple[str, ...]] = ("jpeg", "jpg", "png")
REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0


class ApiClientError(RuntimeError):
    """Raised when the Streamlit UI cannot communicate with the prediction API."""


class JsonHttpResponse(Protocol):
    """Minimal response protocol required for JSON decoding.

    Attributes:
        status: Numeric HTTP status code.
        reason: HTTP reason phrase.
    """

    status: int
    reason: str

    def read(self) -> bytes:
        """Return response body bytes."""


@dataclass(frozen=True)
class PredictionResult:
    """Prediction response shown in the Streamlit UI.

    Attributes:
        predicted_label: Predicted class label returned by the API.
        normal_probability: Probability assigned to the NORMAL class.
        pneumonia_probability: Probability assigned to the PNEUMONIA class.
        confidence: Probability assigned to the predicted class.
        warning: Medical-use warning returned by the API.
    """

    predicted_label: str
    normal_probability: float
    pneumonia_probability: float
    confidence: float
    warning: str


def get_api_base_url() -> str:
    """Return the configured prediction API base URL.

    Returns:
        API base URL from `PNEUMONIA_API_URL`, or the local FastAPI default.
    """

    return os.getenv(API_BASE_URL_ENVIRONMENT_VARIABLE, DEFAULT_API_BASE_URL).rstrip("/")


def fetch_health(api_base_url: str) -> dict[str, object]:
    """Fetch API health status.

    Args:
        api_base_url: Base URL of the FastAPI service.

    Returns:
        Parsed `/health` JSON response.

    Raises:
        ApiClientError: If the request fails or the response is not valid JSON.
    """

    response = send_json_request(api_base_url=api_base_url, path="/health")
    if not isinstance(response, dict):
        message = "The API health response is not a JSON object."
        raise ApiClientError(message)
    return response


def fetch_model_info(api_base_url: str) -> dict[str, object]:
    """Fetch loaded model metadata from the prediction API.

    Args:
        api_base_url: Base URL of the FastAPI service.

    Returns:
        Parsed `/model-info` JSON response.

    Raises:
        ApiClientError: If the request fails or the response is not valid JSON.
    """

    response = send_json_request(api_base_url=api_base_url, path="/model-info")
    if not isinstance(response, dict):
        message = "The API model information response is not a JSON object."
        raise ApiClientError(message)
    return response


def predict_uploaded_image(
    *,
    api_base_url: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> PredictionResult:
    """Send one uploaded image to the prediction API.

    Args:
        api_base_url: Base URL of the FastAPI service.
        image_bytes: Encoded image bytes from the Streamlit uploader.
        filename: Original uploaded filename.
        content_type: MIME content type reported by Streamlit.

    Returns:
        Validated prediction result.

    Raises:
        ApiClientError: If the file is empty, the API request fails, or the response schema is
            invalid.
    """

    if not image_bytes:
        message = "The uploaded image is empty; select a JPEG, JPG, or PNG file."
        raise ApiClientError(message)

    response = send_multipart_image_request(
        api_base_url=api_base_url,
        path="/predict",
        image_bytes=image_bytes,
        filename=filename,
        content_type=content_type,
    )
    if not isinstance(response, dict):
        message = "The API prediction response is not a JSON object."
        raise ApiClientError(message)

    return parse_prediction_response(response)


def send_json_request(*, api_base_url: str, path: str) -> object:
    """Send a JSON GET request to the API.

    Args:
        api_base_url: Base URL of the FastAPI service.
        path: Absolute API path.

    Returns:
        Parsed JSON response.

    Raises:
        ApiClientError: If connection, HTTP status, or JSON parsing fails.
    """

    parsed_url = parse_api_base_url(api_base_url)
    connection = create_connection(parsed_url)
    try:
        request_path = build_request_path(parsed_url, path)
        connection.request("GET", request_path, headers={"Accept": "application/json"})
        response = connection.getresponse()
        return read_json_response(response)
    except (OSError, HTTPException, TimeoutError) as error:
        message = f"API request failed ({api_base_url}{path}): {error}"
        raise ApiClientError(message) from error
    finally:
        connection.close()


def send_multipart_image_request(
    *,
    api_base_url: str,
    path: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> object:
    """Send a multipart image upload request to the API.

    Args:
        api_base_url: Base URL of the FastAPI service.
        path: Absolute API path.
        image_bytes: Encoded image bytes to send as multipart `file`.
        filename: Filename used in the multipart payload.
        content_type: MIME type used in the multipart payload.

    Returns:
        Parsed JSON response.

    Raises:
        ApiClientError: If connection, HTTP status, or JSON parsing fails.
    """

    parsed_url = parse_api_base_url(api_base_url)
    boundary = f"----pneumonia-xray-ui-{uuid.uuid4().hex}"
    body = build_multipart_body(
        boundary=boundary,
        field_name="file",
        filename=filename,
        content_type=content_type,
        payload=image_bytes,
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
    }

    connection = create_connection(parsed_url)
    try:
        request_path = build_request_path(parsed_url, path)
        connection.request("POST", request_path, body=body, headers=headers)
        response = connection.getresponse()
        return read_json_response(response)
    except (OSError, HTTPException, TimeoutError) as error:
        message = f"API prediction request failed ({api_base_url}{path}): {error}"
        raise ApiClientError(message) from error
    finally:
        connection.close()


def parse_api_base_url(api_base_url: str) -> ParseResult:
    """Validate and parse an API base URL.

    Args:
        api_base_url: API base URL entered through configuration.

    Returns:
        Parsed URL object from `urllib.parse.urlparse`.

    Raises:
        ApiClientError: If the URL is missing a supported scheme or host.
    """

    parsed_url = urlparse(api_base_url)
    if parsed_url.scheme not in {"http", "https"}:
        message = (
            f"The API URL must start with `http` or `https`; received: {api_base_url or 'empty'}"
        )
        raise ApiClientError(message)
    if not parsed_url.hostname:
        message = f"The API URL must include a valid host; received: {api_base_url}"
        raise ApiClientError(message)
    return parsed_url


def create_connection(parsed_url: ParseResult) -> HTTPConnection | HTTPSConnection:
    """Create an HTTP connection for a parsed API URL.

    Args:
        parsed_url: URL object returned by `parse_api_base_url`.

    Returns:
        HTTP or HTTPS connection configured with a timeout.
    """

    if parsed_url.scheme == "https":
        return HTTPSConnection(
            parsed_url.hostname,
            port=parsed_url.port,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    return HTTPConnection(
        parsed_url.hostname,
        port=parsed_url.port,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def build_request_path(parsed_url: ParseResult, path: str) -> str:
    """Build a request path that preserves any base URL prefix.

    Args:
        parsed_url: URL object returned by `parse_api_base_url`.
        path: Absolute API path such as `/predict`.

    Returns:
        Request target path including query string when present.
    """

    base_path = parsed_url.path.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    request_path = f"{base_path}{normalized_path}" or "/"
    if parsed_url.query:
        request_path = f"{request_path}?{parsed_url.query}"
    return request_path


def build_multipart_body(
    *,
    boundary: str,
    field_name: str,
    filename: str,
    content_type: str,
    payload: bytes,
) -> bytes:
    """Build a multipart/form-data body containing one image file.

    Args:
        boundary: Multipart boundary without leading dashes.
        field_name: Multipart field name expected by the API.
        filename: Filename to report in `Content-Disposition`.
        content_type: MIME content type for the file part.
        payload: Raw file bytes.

    Returns:
        Encoded multipart request body.
    """

    safe_filename = Path(filename).name or "uploaded_image.png"
    safe_content_type = content_type if content_type.startswith("image/") else "image/png"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{safe_filename}"\r\n'
        f"Content-Type: {safe_content_type}\r\n\r\n"
    ).encode()
    footer = f"\r\n--{boundary}--\r\n".encode()
    return header + payload + footer


def read_json_response(response: JsonHttpResponse) -> object:
    """Read an HTTP response and decode it as JSON.

    Args:
        response: HTTP response returned by `http.client`.

    Returns:
        Parsed JSON payload.

    Raises:
        ApiClientError: If status is not 2xx or the payload is not JSON.
    """

    response_body = response.read()
    text = response_body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text) if text else {}
    except json.JSONDecodeError as error:
        message = f"The API returned a non-JSON response: HTTP {response.status} {response.reason}"
        raise ApiClientError(message) from error

    if response.status < 200 or response.status >= 300:
        detail = extract_error_detail(payload)
        message = f"The API returned an error: HTTP {response.status} {response.reason}. {detail}"
        raise ApiClientError(message)

    return payload


def extract_error_detail(payload: object) -> str:
    """Extract a readable error detail from an API error payload.

    Args:
        payload: Parsed JSON payload returned by the API.

    Returns:
        User-facing detail text.
    """

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    return "No details available."


def parse_prediction_response(payload: dict[str, object]) -> PredictionResult:
    """Validate and convert API prediction JSON into a typed result.

    Args:
        payload: Parsed prediction response from `/predict`.

    Returns:
        Prediction result dataclass.

    Raises:
        ApiClientError: If required fields are missing or invalid.
    """

    predicted_label = payload.get("predicted_label")
    warning = payload.get("warning")
    if not isinstance(predicted_label, str) or predicted_label not in {"NORMAL", "PNEUMONIA"}:
        message = f"The API returned an invalid predicted_label: {predicted_label}"
        raise ApiClientError(message)
    if not isinstance(warning, str) or not warning:
        message = "The API response has a missing or invalid medical warning."
        raise ApiClientError(message)

    return PredictionResult(
        predicted_label=predicted_label,
        normal_probability=parse_probability(payload, "normal_probability"),
        pneumonia_probability=parse_probability(payload, "pneumonia_probability"),
        confidence=parse_probability(payload, "confidence"),
        warning=warning,
    )


def parse_probability(payload: dict[str, object], key: str) -> float:
    """Read and validate a probability value from an API payload.

    Args:
        payload: Parsed prediction response from `/predict`.
        key: Probability field name.

    Returns:
        Probability as a float in the `[0.0, 1.0]` interval.

    Raises:
        ApiClientError: If the value is missing, non-numeric, or out of range.
    """

    raw_value = payload.get(key)
    if not isinstance(raw_value, int | float):
        message = f"`{key}` in the API response is not numeric: {raw_value}"
        raise ApiClientError(message)
    value = float(raw_value)
    if value < 0.0 or value > 1.0:
        message = f"`{key}` in the API response must be between 0 and 1; received: {value}"
        raise ApiClientError(message)
    return value


def render_probability(label: str, probability: float) -> None:
    """Render one probability value as text and a progress bar.

    Args:
        label: Display label for the class probability.
        probability: Probability value in the `[0.0, 1.0]` interval.
    """

    st.write(f"**{label}:** {probability:.2%}")
    st.progress(probability)


def render_sidebar(api_base_url: str) -> None:
    """Render API connection status and startup commands in the sidebar.

    Args:
        api_base_url: Current API base URL.
    """

    st.sidebar.header("Service")
    st.sidebar.caption("PowerShell: select your trained checkpoint, then start the API.")
    st.sidebar.code(
        '$env:PNEUMONIA_MODEL_PATH = "models/clean_runs/selected/best_model.pt"\n'
        "python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000",
        language="powershell",
    )
    st.sidebar.code("python -m streamlit run app/ui/streamlit_app.py")
    st.sidebar.caption(f"API: {api_base_url}")

    if st.sidebar.button("Check API Status"):
        try:
            health = fetch_health(api_base_url)
            model_info = fetch_model_info(api_base_url)
        except ApiClientError as error:
            st.sidebar.error(str(error))
            return

        model_loaded = bool(health.get("model_loaded"))
        status_text = "Model loaded" if model_loaded else "Model not loaded"
        st.sidebar.success(f"API available: {status_text}")
        st.sidebar.write(f"Model: `{model_info.get('model_name', 'unknown')}`")
        st.sidebar.write(f"Checkpoint: `{model_info.get('model_version', 'unknown')}`")


def render_prediction_result(
    result: PredictionResult, decision_threshold: float | None = None
) -> None:
    """Render the prediction result section.

    Args:
        result: Prediction returned by the API.
        decision_threshold: Active API threshold, if available.
    """

    st.subheader("Prediction Result")
    if result.predicted_label == "PNEUMONIA":
        st.error("PNEUMONIA")
    else:
        st.success("NORMAL")

    render_probability("NORMAL probability", result.normal_probability)
    render_probability("PNEUMONIA probability", result.pneumonia_probability)
    st.metric("Selected class probability", f"{result.confidence:.2%}")
    if decision_threshold is not None:
        st.caption(f"PNEUMONIA decision threshold: {decision_threshold:.8f}")
    else:
        st.caption("Active decision threshold is unavailable from the API.")
    st.caption(
        "PNEUMONIA is selected when its probability reaches the decision threshold; "
        "otherwise NORMAL is selected, even if PNEUMONIA has the higher probability. "
        "Selected class probability is the probability assigned to that selected label. "
        "These model scores are not calibrated clinical risk estimates."
    )
    st.warning(result.warning)


def main() -> None:
    """Run the Streamlit application."""

    st.set_page_config(page_title=APP_TITLE, layout="centered")
    api_base_url = get_api_base_url()
    render_sidebar(api_base_url)

    st.title(APP_TITLE)
    st.write(
        "This demo uses a trained CNN to classify an uploaded chest X-ray as "
        "`NORMAL` or `PNEUMONIA`."
    )
    st.warning(MEDICAL_WARNING)

    uploaded_file = st.file_uploader(
        "Upload an X-ray image",
        type=list(SUPPORTED_UPLOAD_TYPES),
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload a chest X-ray in JPEG, JPG, or PNG format to get a prediction.")
        return

    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption=uploaded_file.name, width="stretch")

    if st.button("Predict", type="primary"):
        with st.spinner("Generating prediction..."):
            try:
                result = predict_uploaded_image(
                    api_base_url=api_base_url,
                    image_bytes=image_bytes,
                    filename=uploaded_file.name,
                    content_type=uploaded_file.type or "image/png",
                )
            except ApiClientError as error:
                st.error(str(error))
                st.info(
                    "Make sure the API service is running and the configured checkpoint file "
                    "exists."
                )
                return

        try:
            decision_threshold = parse_probability(
                fetch_model_info(api_base_url), "decision_threshold"
            )
        except ApiClientError:
            decision_threshold = None
        render_prediction_result(result, decision_threshold)


if __name__ == "__main__":
    main()
