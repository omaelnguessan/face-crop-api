from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.crop import Crop
from app.detector import FaceBox
from app.main import app, build_openinary_url, get_service
from app.service import Detection, SourceError

SRC = (
    "https://openinary.icoop.live/t/www-idiscover-live/image/upload/"
    "v1708680528/mob_avatar/2103549847_2026-05-05_12-16-10.jpg"
)


class FakeService:
    """Service mocké : ni réseau, ni modèle ONNX."""

    def __init__(self) -> None:
        self.detection = Detection(
            width=1200,
            height=800,
            faces=(FaceBox(x=500, y=200, w=200, h=200, score=0.95),),
        )
        self.payload = b"\xff\xd8\xff\xe0 fake jpeg bytes"
        self.fail: Exception | None = None

    async def detect(self, url: str) -> Detection:
        if self.fail:
            raise self.fail
        return self.detection

    async def render(self, url, w, h, zoom, all_faces, quality) -> bytes:
        if self.fail:
            raise self.fail
        return self.payload


@pytest.fixture()
def fake() -> FakeService:
    return FakeService()


@pytest.fixture()
def client(fake: FakeService):
    app.dependency_overrides[get_service] = lambda: fake
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").status_code == 200


def test_coords_returns_crop(client: TestClient) -> None:
    response = client.get("/v1/face/coords", params={"url": SRC, "w": 200, "h": 200})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == {"width": 1200, "height": 800}
    assert body["faces"] == 1
    crop = body["crop"]
    assert crop["x"] + crop["w"] <= 1200
    assert crop["y"] + crop["h"] <= 800
    assert body["transformation"].startswith("c_crop,x_")


def test_host_not_allowlisted_is_403(client: TestClient) -> None:
    response = client.get(
        "/v1/face/coords", params={"url": "https://evil.example.com/a.jpg"}
    )
    assert response.status_code == 403
    assert "hôte non autorisé" in response.json()["detail"]


def test_http_scheme_is_403(client: TestClient) -> None:
    response = client.get(
        "/v1/face/coords",
        params={"url": SRC.replace("https://", "http://")},
    )
    assert response.status_code == 403


def test_ssrf_lookalike_host_is_403(client: TestClient) -> None:
    # L'hôte réel est `attacker.tld`, pas le domaine autorisé présent dans le path.
    sneaky = "https://attacker.tld/openinary.icoop.live/image/upload/v1/a.jpg"
    assert client.get("/v1/face/coords", params={"url": sneaky}).status_code == 403


@pytest.mark.parametrize("dims", [{"w": 4}, {"h": 5000}, {"w": 0, "h": 0}])
def test_out_of_range_dimensions_are_400(client: TestClient, dims: dict) -> None:
    response = client.get("/v1/face/coords", params={"url": SRC, **dims})
    assert response.status_code == 400


def test_redirect_url_construction(client: TestClient) -> None:
    response = client.get(
        "/v1/face/redirect",
        params={"url": SRC, "w": 200, "h": 200},
        follow_redirects=False,
    )
    assert response.status_code == 302
    location = response.headers["location"]
    assert location.startswith(
        "https://openinary.icoop.live/t/www-idiscover-live/image/upload/c_crop,"
    )
    assert "/c_fill,w_200,h_200/v1708680528/mob_avatar/" in location
    assert location.endswith("2103549847_2026-05-05_12-16-10.jpg")
    assert response.headers["cache-control"] == "public, max-age=604800"


def test_build_openinary_url_is_exact() -> None:
    url = build_openinary_url(SRC, Crop(x=312, y=98, w=540, h=540), 200, 200)
    assert url == (
        "https://openinary.icoop.live/t/www-idiscover-live/image/upload/"
        "c_crop,x_312,y_98,w_540,h_540/c_fill,w_200,h_200/"
        "v1708680528/mob_avatar/2103549847_2026-05-05_12-16-10.jpg"
    )


def test_render_headers(client: TestClient, fake: FakeService) -> None:
    response = client.get("/v1/face/render", params={"url": SRC, "w": 200, "h": 200})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=2592000"
    assert response.headers["etag"] == f'"{hashlib.md5(fake.payload).hexdigest()}"'


def test_source_error_becomes_422(client: TestClient, fake: FakeService) -> None:
    fake.fail = SourceError("format d'image non décodable")
    response = client.get("/v1/face/coords", params={"url": SRC})
    assert response.status_code == 422
    assert response.json()["detail"] == "format d'image non décodable"


def test_url_without_upload_segment_is_422(client: TestClient) -> None:
    response = client.get(
        "/v1/face/coords",
        params={"url": "https://openinary.icoop.live/static/avatar.jpg"},
    )
    assert response.status_code == 422


def test_preset_id_sets_dimensions_and_zoom(client: TestClient) -> None:
    response = client.get("/v1/face/coords", params={"url": SRC, "preset": "id"})
    assert response.status_code == 200
    body = response.json()
    crop = body["crop"]
    # Ratio 35x45 conservé par le cadrage.
    assert abs(crop["w"] / crop["h"] - 413 / 531) < 0.01
    assert "c_fill,w_413,h_531" in body["url"]
    assert body["preset"] == "id"


def test_preset_id_clean_source_has_no_warnings(
    client: TestClient, fake: FakeService
) -> None:
    # Visage assez grand pour que le crop dépasse 413x531 sans agrandissement.
    fake.detection = Detection(
        width=2000,
        height=2500,
        faces=(FaceBox(x=700, y=600, w=500, h=600, score=0.97),),
    )
    body = client.get(
        "/v1/face/coords", params={"url": SRC, "preset": "id"}
    ).json()
    assert body["warnings"] == []


def test_preset_id_warns_on_multiple_faces(
    client: TestClient, fake: FakeService
) -> None:
    fake.detection = Detection(
        width=1200,
        height=800,
        faces=(
            FaceBox(x=500, y=200, w=200, h=200, score=0.95),
            FaceBox(x=100, y=200, w=150, h=150, score=0.92),
        ),
    )
    body = client.get(
        "/v1/face/coords", params={"url": SRC, "preset": "id"}
    ).json()
    assert any("2 visages" in w for w in body["warnings"])


def test_preset_id_warns_when_no_face(client: TestClient, fake: FakeService) -> None:
    fake.detection = Detection(width=1200, height=800, faces=())
    body = client.get(
        "/v1/face/coords", params={"url": SRC, "preset": "id"}
    ).json()
    assert any("aucun visage" in w for w in body["warnings"])


def test_preset_id_warns_on_low_resolution(
    client: TestClient, fake: FakeService
) -> None:
    fake.detection = Detection(
        width=300,
        height=300,
        faces=(FaceBox(x=100, y=80, w=90, h=90, score=0.95),),
    )
    body = client.get(
        "/v1/face/coords", params={"url": SRC, "preset": "id"}
    ).json()
    assert any("résolution source insuffisante" in w for w in body["warnings"])


def test_explicit_params_override_preset(client: TestClient) -> None:
    body = client.get(
        "/v1/face/coords", params={"url": SRC, "preset": "id", "w": 200, "h": 200}
    ).json()
    assert "c_fill,w_200,h_200" in body["url"]


def test_unknown_preset_is_400(client: TestClient) -> None:
    response = client.get("/v1/face/coords", params={"url": SRC, "preset": "passport"})
    assert response.status_code == 400
    assert "preset inconnu" in response.json()["detail"]


def test_no_preset_keeps_400x400_default(client: TestClient) -> None:
    body = client.get("/v1/face/coords", params={"url": SRC}).json()
    assert "c_fill,w_400,h_400" in body["url"]
    assert "warnings" not in body
