from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path

import pytest
from app.ui import streamlit_app
from app.ui.streamlit_app import (
    ApiClientError,
    build_multipart_body,
    build_request_path,
    parse_api_base_url,
    parse_prediction_response,
    read_json_response,
)


def test_build_request_path_preserves_base_prefix_and_query() -> None:
    parsed_url = parse_api_base_url("http://127.0.0.1:8000/api?debug=true")

    request_path = build_request_path(parsed_url, "/predict")

    assert request_path == "/api/predict?debug=true"


def test_build_multipart_body_contains_file_part(tmp_path: Path) -> None:
    payload = b"image bytes"
    body = build_multipart_body(
        boundary="boundary",
        field_name="file",
        filename=str(tmp_path / "sample.png"),
        content_type="image/png",
        payload=payload,
    )

    assert b"--boundary" in body
    assert b'name="file"; filename="sample.png"' in body
    assert b"Content-Type: image/png" in body
    assert payload in body


def test_parse_prediction_response_returns_typed_result() -> None:
    result = parse_prediction_response(
        {
            "predicted_label": "PNEUMONIA",
            "normal_probability": 0.2,
            "pneumonia_probability": 0.8,
            "confidence": 0.8,
            "warning": "This output must not be used for medical diagnosis.",
        }
    )

    assert result.predicted_label == "PNEUMONIA"
    assert result.normal_probability == 0.2
    assert result.pneumonia_probability == 0.8
    assert result.confidence == 0.8


def test_parse_prediction_response_rejects_invalid_probability() -> None:
    with pytest.raises(ApiClientError, match="between 0 and 1"):
        parse_prediction_response(
            {
                "predicted_label": "NORMAL",
                "normal_probability": 1.2,
                "pneumonia_probability": 0.1,
                "confidence": 1.2,
                "warning": "This output must not be used for medical diagnosis.",
            }
        )


def test_parse_api_base_url_rejects_unsupported_scheme() -> None:
    with pytest.raises(ApiClientError, match="http` or `https"):
        parse_api_base_url("ftp://127.0.0.1:8000")


@pytest.mark.parametrize("threshold", [0.88200253, None])
def test_result_explains_threshold_selected_class(
    monkeypatch: pytest.MonkeyPatch, threshold: float | None
) -> None:
    calls = []
    methods = ("subheader", "success", "error", "write", "progress", "metric", "caption", "warning")
    for name in methods:
        monkeypatch.setattr(streamlit_app.st, name, lambda *args: calls.append(args))
    result = parse_prediction_response(
        {
            "predicted_label": "NORMAL",
            "normal_probability": 0.2,
            "pneumonia_probability": 0.8,
            "confidence": 0.2,
            "warning": "Research only",
        }
    )
    streamlit_app.render_prediction_result(result, threshold)
    assert ("NORMAL",) in calls
    assert ("Selected class probability", "20.00%") in calls
    captions = " ".join(str(call) for call in calls)
    assert "not calibrated clinical risk estimates" in captions
    assert ("0.88200253" if threshold is not None else "unavailable") in captions


def test_read_json_response_raises_readable_api_error() -> None:
    response = _FakeHTTPResponse(
        status=HTTPStatus.BAD_REQUEST,
        reason="Bad Request",
        body=json.dumps({"detail": "Invalid image"}).encode("utf-8"),
    )

    with pytest.raises(ApiClientError, match="Invalid image"):
        read_json_response(response)


class _FakeHTTPResponse:
    def __init__(self, *, status: int, reason: str, body: bytes) -> None:
        self.status = status
        self.reason = reason
        self._body = body

    def read(self) -> bytes:
        return self._body
