from __future__ import annotations

import base64
import json
import stat
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from quant_platform.cli.schwab_auth import (
    SchwabAuthError,
    build_authorization_url,
    exchange_authorization_code,
    parse_authorization_code,
    run_bootstrap,
    write_token_file,
)
from quant_platform.config.settings import Settings


class MockResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> MockResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        schwab_client_id="client-id",
        schwab_client_secret="client-secret",
        schwab_redirect_uri="https://127.0.0.1:8182/callback",
        schwab_token_path=tmp_path / "secrets" / "tokens.json",
        schwab_api_timeout_seconds=7.5,
    )


def test_build_authorization_url_uses_configured_settings(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    url = build_authorization_url(settings)

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "api.schwabapi.com"
    assert parsed.path == "/v1/oauth/authorize"
    assert query == {
        "response_type": ["code"],
        "client_id": ["client-id"],
        "redirect_uri": ["https://127.0.0.1:8182/callback"],
    }


@pytest.mark.parametrize(
    ("pasted_value", "expected_code"),
    [
        ("https://127.0.0.1:8182/callback?code=abc123&state=ignored", "abc123"),
        (" raw-code ", "raw-code"),
    ],
)
def test_parse_authorization_code_accepts_callback_url_or_raw_code(
    pasted_value: str, expected_code: str
) -> None:
    assert parse_authorization_code(pasted_value) == expected_code


def test_parse_authorization_code_rejects_callback_errors() -> None:
    with pytest.raises(SchwabAuthError, match="error"):
        parse_authorization_code("https://127.0.0.1:8182/callback?error=denied")


def test_exchange_authorization_code_posts_expected_request(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    captured = {}

    def opener(request, timeout: float):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["data"] = request.data.decode()
        return MockResponse(
            {
                "access_token": "access-token-value",
                "refresh_token": "refresh-token-value",
                "expires_in": 1800,
            }
        )

    tokens = exchange_authorization_code(settings, "auth-code", opener=opener)

    assert tokens["expires_in"] == 1800
    assert captured["url"] == "https://api.schwabapi.com/v1/oauth/token"
    assert captured["timeout"] == 7.5
    expected_auth = base64.b64encode(b"client-id:client-secret").decode()
    assert captured["headers"]["Authorization"] == f"Basic {expected_auth}"
    assert parse_qs(captured["data"]) == {
        "grant_type": ["authorization_code"],
        "code": ["auth-code"],
        "redirect_uri": ["https://127.0.0.1:8182/callback"],
    }


def test_write_token_file_uses_restrictive_permissions(tmp_path: Path) -> None:
    token_path = tmp_path / "nested" / "schwab_tokens.json"
    tokens = {"access_token": "access-token", "refresh_token": "refresh-token"}

    write_token_file(tokens, token_path)

    assert json.loads(token_path.read_text()) == tokens
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_run_bootstrap_never_prints_or_logs_tokens_or_client_secret(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path)
    outputs: list[str] = []
    secret_values = [
        "client-secret",
        "access-token-value",
        "refresh-token-value",
        "123456789",
    ]

    def fake_exchange(_settings: Settings, code: str):  # type: ignore[no-untyped-def]
        assert code == "auth-code"
        return {
            "access_token": "access-token-value",
            "refresh_token": "refresh-token-value",
            "accountNumber": "123456789",
        }

    monkeypatch.setattr(
        "quant_platform.cli.schwab_auth.exchange_authorization_code", fake_exchange
    )

    run_bootstrap(
        settings,
        input_fn=lambda _prompt: "https://127.0.0.1:8182/callback?code=auth-code",
        print_fn=lambda *parts, **_kwargs: outputs.append(" ".join(map(str, parts))),
    )

    printed = "\n".join(outputs)
    for secret_value in secret_values:
        assert secret_value not in printed
    assert settings.schwab_token_path.read_text().count("access-token-value") == 1
    assert stat.S_IMODE(settings.schwab_token_path.stat().st_mode) == 0o600
