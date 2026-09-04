from __future__ import annotations

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch, tmp_path):
    """Isole les tests d'un éventuel `.env` du dépôt."""
    monkeypatch.chdir(tmp_path)


def test_defaults_without_env() -> None:
    assert Settings().allowed_hosts == ("openinary.icoop.live",)


def test_allowed_hosts_from_env_single(monkeypatch) -> None:
    """Régression : une valeur non-JSON faisait échouer le démarrage."""
    monkeypatch.setenv("ALLOWED_HOSTS", "openinary.icoop.live")
    assert Settings().allowed_hosts == ("openinary.icoop.live",)


def test_allowed_hosts_from_env_comma_separated(monkeypatch) -> None:
    monkeypatch.setenv("ALLOWED_HOSTS", "openinary.icoop.live, CDN.example.com ")
    assert Settings().allowed_hosts == ("openinary.icoop.live", "cdn.example.com")


def test_other_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_ZOOM", "1.9")
    monkeypatch.setenv("WORKERS", "2")
    settings = Settings()
    assert settings.default_zoom == 1.9
    assert settings.workers == 2
