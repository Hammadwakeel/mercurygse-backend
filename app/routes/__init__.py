from fastapi import APIRouter

router = APIRouter()

from . import process, health  # noqa: E402,F401

router.include_router(process.router, prefix="/process", tags=["process"])
router.include_router(health.router, prefix="/health", tags=["health"])
