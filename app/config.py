from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration du service, surchargeable par variables d'environnement."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    allowed_hosts: tuple[str, ...] = ("openinary.icoop.live",)
    max_dimension: int = 2000
    min_dimension: int = 16
    max_bytes: int = 12 * 1024 * 1024
    score_threshold: float = 0.75
    nms_threshold: float = 0.3
    default_zoom: float = 2.6
    workers: int = 4
    cache_ttl: int = 7 * 24 * 3600
    cache_max_entries: int = 200_000
    model_path: str = "models/face_detection_yunet_2023mar.onnx"
    detect_max_side: int = 1024
    fetch_timeout: float = 5.0

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _split_hosts(cls, value: object) -> object:
        """Accepte `ALLOWED_HOSTS=a.com,b.com` aussi bien qu'une liste."""
        if isinstance(value, str):
            return tuple(h.strip().lower() for h in value.split(",") if h.strip())
        if isinstance(value, (list, tuple)):
            return tuple(str(h).strip().lower() for h in value)
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Instance unique de configuration (mise en cache)."""
    return Settings()
