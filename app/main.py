from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import urlparse, urlunparse

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

from .config import Settings, get_settings
from .crop import Crop, plan
from .presets import PRESETS, id_warnings, resolve as resolve_preset
from .service import FaceService, SourceError

logger = logging.getLogger("face_crop_api")

RENDER_CACHE_CONTROL = "public, max-age=2592000"
REDIRECT_CACHE_CONTROL = "public, max-age=604800"


@dataclass(frozen=True, slots=True)
class Params:
    """Paramètres validés par le garde-fou, sûrs à utiliser côté réseau."""

    url: str
    w: int
    h: int
    zoom: float
    all_faces: bool
    preset: str | None


def guard(
    url: str = Query(..., description="URL https de l'image source"),
    w: int | None = Query(None, description="Largeur cible (défaut 400)"),
    h: int | None = Query(None, description="Hauteur cible (défaut 400)"),
    zoom: float | None = Query(None, description="Hauteur du crop en hauteurs de visage"),
    all_faces: bool = Query(False, description="Englober tous les visages détectés"),
    preset: str | None = Query(
        None, description=f"Preset de cadrage : {', '.join(PRESETS)}"
    ),
    settings: Settings = Depends(get_settings),
) -> Params:
    """Valide l'URL et les dimensions avant tout accès réseau (anti-SSRF)."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise HTTPException(status_code=403, detail="seul le schéma https est autorisé")

    hostname = (parsed.hostname or "").lower()
    if hostname not in settings.allowed_hosts:
        raise HTTPException(status_code=403, detail=f"hôte non autorisé : {hostname or '?'}")

    try:
        chosen = resolve_preset(preset)
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"preset inconnu : {preset} (disponibles : {', '.join(PRESETS)})",
        ) from exc

    # Un paramètre explicite l'emporte toujours sur la valeur du preset.
    resolved_w = w if w is not None else (chosen.w if chosen else 400)
    resolved_h = h if h is not None else (chosen.h if chosen else 400)
    resolved_zoom = (
        zoom
        if zoom is not None
        else (chosen.zoom if chosen else settings.default_zoom)
    )

    lo, hi = settings.min_dimension, settings.max_dimension
    if not (lo <= resolved_w <= hi) or not (lo <= resolved_h <= hi):
        raise HTTPException(
            status_code=400, detail=f"dimensions hors bornes ({lo}–{hi})"
        )

    if not (1.0 <= resolved_zoom <= 12.0):
        raise HTTPException(status_code=400, detail="zoom hors bornes (1.0–12.0)")

    return Params(
        url=url,
        w=resolved_w,
        h=resolved_h,
        zoom=resolved_zoom,
        all_faces=all_faces,
        preset=preset.strip().lower() if preset else None,
    )


def build_openinary_url(source_url: str, crop: Crop, w: int, h: int) -> str:
    """Insère `c_crop,…/c_fill,w_,h_` juste après le segment `upload` de l'URL."""
    parsed = urlparse(source_url)
    segments = parsed.path.split("/")
    try:
        index = len(segments) - 1 - segments[::-1].index("upload")
    except ValueError as exc:  # pas une URL de livraison Openinary/Cloudinary
        raise HTTPException(
            status_code=422, detail="URL source sans segment `upload`"
        ) from exc

    transformation = [crop.to_openinary(), f"c_fill,w_{w},h_{h}"]
    new_path = "/".join(segments[: index + 1] + transformation + segments[index + 1 :])
    return urlunparse(parsed._replace(path=new_path))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    service: FaceService | None = None
    try:
        service = FaceService(settings)
    except FileNotFoundError as exc:
        logger.warning("Service dégradé : %s", exc)
    app.state.service = service
    try:
        yield
    finally:
        if service is not None:
            await service.aclose()
        app.state.service = None


app = FastAPI(
    title="face-crop-api",
    version="1.0.0",
    summary="Coordonnées de crop centré visage pour URLs Openinary",
    lifespan=lifespan,
)


def get_service(request: Request) -> FaceService:
    """Dépendance : le service initialisé dans le lifespan (surchargeable en test)."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="modèle de détection indisponible")
    return service


@app.exception_handler(SourceError)
async def _source_error_handler(_: Request, exc: SourceError) -> JSONResponse:
    """Les problèmes d'image source deviennent des 422 lisibles, jamais une stacktrace."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/healthz", summary="Sonde de vivacité")
async def healthz(request: Request) -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": getattr(request.app.state, "service", None) is not None,
    }


@app.get("/v1/face/coords", summary="Coordonnées du crop")
async def coords(
    params: Params = Depends(guard),
    service: FaceService = Depends(get_service),
) -> dict[str, object]:
    detection = await service.detect(params.url)
    rect = plan(
        detection.width,
        detection.height,
        detection.faces,
        params.w,
        params.h,
        zoom=params.zoom,
        all_faces=params.all_faces,
    )
    payload: dict[str, object] = {
        "source": {"width": detection.width, "height": detection.height},
        "faces": len(detection.faces),
        "crop": rect.as_dict(),
        "transformation": rect.to_openinary(),
        "url": build_openinary_url(params.url, rect, params.w, params.h),
    }
    if params.preset == "id":
        payload["preset"] = "id"
        payload["warnings"] = id_warnings(
            detection.faces, rect, params.w, params.h, params.zoom
        )
    return payload


@app.get("/v1/face/redirect", summary="Redirection vers l'URL Openinary recadrée")
async def redirect(
    params: Params = Depends(guard),
    service: FaceService = Depends(get_service),
) -> RedirectResponse:
    detection = await service.detect(params.url)
    rect = plan(
        detection.width,
        detection.height,
        detection.faces,
        params.w,
        params.h,
        zoom=params.zoom,
        all_faces=params.all_faces,
    )
    target = build_openinary_url(params.url, rect, params.w, params.h)
    return RedirectResponse(
        url=target,
        status_code=302,
        headers={"Cache-Control": REDIRECT_CACHE_CONTROL},
    )


@app.get("/v1/face/render", summary="JPEG recadré (mode proxy)")
async def render(
    params: Params = Depends(guard),
    quality: int = Query(82, ge=1, le=100),
    service: FaceService = Depends(get_service),
) -> Response:
    payload = await service.render(
        params.url, params.w, params.h, params.zoom, params.all_faces, quality
    )
    return Response(
        content=payload,
        media_type="image/jpeg",
        headers={
            "Cache-Control": RENDER_CACHE_CONTROL,
            "ETag": f'"{hashlib.md5(payload).hexdigest()}"',
        },
    )
