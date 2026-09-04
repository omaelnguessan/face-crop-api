from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import httpx
import numpy as np

from .cache import TTLCache
from .config import Settings
from .crop import Crop, plan
from .detector import FaceBox, YuNetDetector


class SourceError(Exception):
    """Erreur imputable à l'image source (téléchargement ou décodage) → HTTP 422."""


@dataclass(frozen=True, slots=True)
class Detection:
    """Résultat de détection mis en cache : dimensions + visages, jamais les pixels."""

    width: int
    height: int
    faces: tuple[FaceBox, ...]


class FaceService:
    """Orchestration : téléchargement, décodage, détection, rendu."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.fetch_timeout),
            follow_redirects=False,
            headers={"User-Agent": "face-crop-api/1.0"},
        )
        self._pool = ThreadPoolExecutor(
            max_workers=settings.workers, thread_name_prefix="face-cv"
        )
        self._cache: TTLCache[Detection] = TTLCache(
            ttl=settings.cache_ttl, max_entries=settings.cache_max_entries
        )
        self._detector = YuNetDetector(
            model_path=settings.model_path,
            score_threshold=settings.score_threshold,
            nms_threshold=settings.nms_threshold,
            max_side=settings.detect_max_side,
        )

    # ------------------------------------------------------------------ fetch

    async def _fetch(self, url: str) -> bytes:
        """Télécharge en streaming, avec plafond strict : on coupe dès dépassement."""
        limit = self._settings.max_bytes
        chunks: list[bytes] = []
        total = 0
        try:
            async with self._client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise SourceError(
                        f"source indisponible (HTTP {response.status_code})"
                    )
                if response.status_code >= 300:
                    raise SourceError("redirection refusée sur l'URL source")

                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > limit:
                    raise SourceError(f"image trop volumineuse (> {limit} octets)")

                async for chunk in response.aiter_bytes(64 * 1024):
                    total += len(chunk)
                    if total > limit:
                        raise SourceError(f"image trop volumineuse (> {limit} octets)")
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise SourceError(f"téléchargement impossible : {exc.__class__.__name__}") from exc

        if not chunks:
            raise SourceError("réponse source vide")
        return b"".join(chunks)

    # ----------------------------------------------------------------- CPU

    def _decode(self, payload: bytes) -> np.ndarray:
        """Décode les octets en image BGR (bloquant : à exécuter hors boucle d'événements)."""
        buffer = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise SourceError("format d'image non décodable")
        return image

    def _decode_and_detect(self, payload: bytes) -> tuple[np.ndarray, Detection]:
        image = self._decode(payload)
        height, width = image.shape[:2]
        faces = tuple(self._detector.detect(image))
        return image, Detection(width=width, height=height, faces=faces)

    async def _run(self, func, *args):  # type: ignore[no-untyped-def]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, func, *args)

    # ------------------------------------------------------------- opérations

    async def detect(self, url: str) -> Detection:
        """Dimensions + visages de l'image source, avec cache par URL."""
        cached = self._cache.get(url)
        if cached is not None:
            return cached

        payload = await self._fetch(url)
        _, detection = await self._run(self._decode_and_detect, payload)
        self._cache.set(url, detection)
        return detection

    async def render(
        self,
        url: str,
        target_w: int,
        target_h: int,
        zoom: float,
        all_faces: bool,
        quality: int,
    ) -> bytes:
        """Retourne les octets JPEG de l'image recadrée puis redimensionnée."""
        payload = await self._fetch(url)
        image, detection = await self._run(self._decode_and_detect, payload)
        self._cache.set(url, detection)

        crop = plan(
            detection.width,
            detection.height,
            detection.faces,
            target_w,
            target_h,
            zoom=zoom,
            all_faces=all_faces,
        )
        return await self._run(
            self._encode, image, crop, target_w, target_h, quality
        )

    def _encode(
        self,
        image: np.ndarray,
        crop: Crop,
        target_w: int,
        target_h: int,
        quality: int,
    ) -> bytes:
        patch = image[crop.y : crop.y + crop.h, crop.x : crop.x + crop.w]
        if patch.size == 0:
            raise SourceError("crop vide")
        interpolation = (
            cv2.INTER_AREA
            if (target_w < crop.w or target_h < crop.h)
            else cv2.INTER_CUBIC
        )
        resized = cv2.resize(patch, (target_w, target_h), interpolation=interpolation)
        ok, encoded = cv2.imencode(
            ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        )
        if not ok:
            raise SourceError("encodage JPEG impossible")
        return encoded.tobytes()

    # ------------------------------------------------------------------ cycle

    async def aclose(self) -> None:
        """Fermeture propre du client httpx et du pool de threads."""
        await self._client.aclose()
        self._pool.shutdown(wait=True, cancel_futures=True)
