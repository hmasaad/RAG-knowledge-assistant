"""FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends, Header, Request

from app.security.access import normalize_user_id
from app.wiring import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_user_id(
    services: Services = Depends(get_services),
    x_user_id: Annotated[str | None, Header()] = None,
) -> str:
    return normalize_user_id(x_user_id, default=services.settings.default_user_id)
