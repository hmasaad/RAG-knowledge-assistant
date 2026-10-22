"""FastAPI dependencies."""

from fastapi import Request

from app.wiring import Services


def get_services(request: Request) -> Services:
    return request.app.state.services
