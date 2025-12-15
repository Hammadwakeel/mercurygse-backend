from fastapi import APIRouter

router = APIRouter()


@router.get("/live")
async def live():
    return {"status": "ok"}


@router.get("/ready")
async def ready():
    return {"status": "ready"}
