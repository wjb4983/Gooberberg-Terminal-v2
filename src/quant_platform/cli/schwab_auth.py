"""Bootstrap Schwab OAuth tokens without printing secrets."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from quant_platform.config.settings import Settings, get_settings

SCHWAB_AUTHORIZE_URL = "https://api.schwabapi.com/v1/oauth/authorize"
SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

InputFn = Callable[[str], str]
PrintFn = Callable[..., None]


class SchwabAuthError(RuntimeError):
    """Raised when the Schwab OAuth bootstrap flow cannot complete."""


def _require_setting(value: str | None, name: str) -> str:
    if not value:
        raise SchwabAuthError(f"Missing required setting: {name}")
    return value


def build_authorization_url(
    settings: Settings,
    *,
    authorize_url: str = SCHWAB_AUTHORIZE_URL,
) -> str:
    """Build the Schwab authorization URL from configured settings."""

    client_id = _require_setting(settings.schwab_client_id, "schwab_client_id")
    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": str(settings.schwab_redirect_uri),
        }
    )
    return f"{authorize_url}?{query}"


def parse_authorization_code(value: str) -> str:
    """Return an authorization code from a pasted callback URL or raw code."""

    candidate = value.strip()
    if not candidate:
        raise SchwabAuthError("Authorization callback URL or code cannot be empty.")

    parsed = urlparse(candidate)
    if parsed.scheme and parsed.netloc:
        params = parse_qs(parsed.query, keep_blank_values=False)
        errors = params.get("error")
        if errors:
            raise SchwabAuthError("Schwab authorization callback contained an error.")
        codes = params.get("code")
        if not codes or not codes[0].strip():
            raise SchwabAuthError(
                "Schwab authorization callback did not include a code."
            )
        return codes[0].strip()

    return candidate


def exchange_authorization_code(
    settings: Settings,
    code: str,
    *,
    token_url: str = SCHWAB_TOKEN_URL,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Exchange a Schwab authorization code for tokens."""

    client_id = _require_setting(settings.schwab_client_id, "schwab_client_id")
    client_secret = _require_setting(
        settings.schwab_client_secret, "schwab_client_secret"
    )
    basic_token = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    body = urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": str(settings.schwab_redirect_uri),
        }
    ).encode()
    request = Request(
        token_url,
        data=body,
        headers={
            "Authorization": f"Basic {basic_token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with opener(request, timeout=settings.schwab_api_timeout_seconds) as response:
            payload = response.read()
    except HTTPError as exc:
        raise SchwabAuthError(
            "Schwab token exchange failed with an HTTP error."
        ) from exc
    except URLError as exc:
        raise SchwabAuthError(
            "Schwab token exchange failed with a network error."
        ) from exc

    try:
        tokens = json.loads(payload.decode())
    except json.JSONDecodeError as exc:
        raise SchwabAuthError("Schwab token response was not valid JSON.") from exc

    if not isinstance(tokens, dict):
        raise SchwabAuthError("Schwab token response was not a JSON object.")
    if not tokens.get("access_token") or not tokens.get("refresh_token"):
        raise SchwabAuthError("Schwab token response did not include expected tokens.")
    return tokens


def write_token_file(tokens: Mapping[str, Any], token_path: Path) -> None:
    """Write tokens as JSON with owner-only file permissions."""

    token_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(token_path, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as token_file:
        json.dump(dict(tokens), token_file, indent=2, sort_keys=True)
        token_file.write("\n")
    os.chmod(token_path, 0o600)


def run_bootstrap(
    settings: Settings,
    *,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
) -> None:
    """Run the interactive Schwab OAuth bootstrap flow."""

    authorization_url = build_authorization_url(settings)
    print_fn("Open this Schwab authorization URL in a browser:")
    print_fn(authorization_url)
    pasted_value = input_fn("Paste the redirected callback URL or authorization code: ")
    code = parse_authorization_code(pasted_value)
    tokens = exchange_authorization_code(settings, code)
    write_token_file(tokens, settings.schwab_token_path)
    print_fn(f"Schwab token file written to {settings.schwab_token_path}")


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Build a Schwab authorization URL and save OAuth tokens."
    )


def main(argv: Sequence[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    try:
        run_bootstrap(get_settings())
    except SchwabAuthError as exc:
        print(f"Schwab authorization failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
