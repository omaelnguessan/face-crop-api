from __future__ import annotations

import pytest

from app.crop import plan
from app.detector import FaceBox


def _inside(crop, img_w: int, img_h: int) -> bool:
    return (
        crop.x >= 0
        and crop.y >= 0
        and crop.w > 0
        and crop.h > 0
        and crop.x + crop.w <= img_w
        and crop.y + crop.h <= img_h
    )


def test_no_face_returns_centered_crop() -> None:
    crop = plan(1000, 600, [], 400, 400)
    assert _inside(crop, 1000, 600)
    assert crop.w == crop.h == 600
    assert crop.x == 200 and crop.y == 0


def test_no_face_non_square_ratio() -> None:
    crop = plan(800, 800, [], 400, 200)
    assert _inside(crop, 800, 800)
    assert crop.w == 800 and crop.h == 400
    assert crop.y == 200  # centré verticalement


def test_crop_is_centered_on_face_and_lifted() -> None:
    face = FaceBox(x=400, y=300, w=200, h=200, score=0.9)
    crop = plan(1200, 1200, [face], 400, 400)
    assert _inside(crop, 1200, 1200)
    # Centré horizontalement sur le visage.
    assert abs((crop.x + crop.w / 2) - face.center_x) <= 1
    # Le centre du crop est au-dessus du centre du visage (air au-dessus de la tête).
    assert crop.y + crop.h / 2 < face.center_y


def test_zoom_controls_crop_height() -> None:
    face = FaceBox(x=500, y=500, w=100, h=100, score=0.9)
    tight = plan(2000, 2000, [face], 400, 400, zoom=2.6)
    wide = plan(2000, 2000, [face], 400, 400, zoom=5.0)
    assert tight.h == 260
    assert wide.h == 500
    assert wide.h > tight.h


def test_face_fits_in_width() -> None:
    # Visage très large et très plat : la contrainte de largeur doit primer.
    face = FaceBox(x=100, y=400, w=800, h=100, score=0.9)
    crop = plan(2000, 2000, [face], 400, 400)
    assert crop.w >= 1.25 * face.w
    assert _inside(crop, 2000, 2000)


@pytest.mark.parametrize(
    ("img_w", "img_h", "fx", "fy", "fw", "fh", "tw", "th", "zoom"),
    [
        (400, 400, 0, 0, 60, 60, 400, 400, 2.6),          # visage collé au coin
        (400, 400, 340, 340, 60, 60, 400, 400, 4.0),      # coin opposé
        (1920, 1080, 900, 100, 300, 300, 200, 200, 3.0),  # paysage
        (600, 1600, 250, 1400, 120, 120, 300, 100, 2.0),  # portrait, visage bas
        (50, 50, 10, 10, 30, 30, 2000, 2000, 6.0),        # cible plus grande que la source
        (1000, 1000, 400, 400, 200, 200, 16, 2000, 2.6),  # ratio extrême
    ],
)
def test_crop_always_inside_bounds(
    img_w: int, img_h: int, fx: int, fy: int, fw: int, fh: int,
    tw: int, th: int, zoom: float,
) -> None:
    face = FaceBox(x=fx, y=fy, w=fw, h=fh, score=0.9)
    crop = plan(img_w, img_h, [face], tw, th, zoom=zoom)
    assert _inside(crop, img_w, img_h)


@pytest.mark.parametrize(("tw", "th"), [(400, 400), (300, 100), (100, 300), (1600, 900)])
def test_crop_respects_target_ratio(tw: int, th: int) -> None:
    face = FaceBox(x=800, y=700, w=200, h=200, score=0.9)
    crop = plan(3000, 3000, [face], tw, th)
    assert crop.w / crop.h == pytest.approx(tw / th, rel=0.02)


def test_all_faces_expands_to_union() -> None:
    faces = [
        FaceBox(x=200, y=900, w=200, h=200, score=0.9),
        FaceBox(x=1400, y=920, w=180, h=180, score=0.8),
    ]
    single = plan(2400, 2400, faces, 400, 400, all_faces=False)
    group = plan(2400, 2400, faces, 400, 400, all_faces=True)
    assert group.w > single.w
    # La boîte union tient dans le crop : les deux visages sont cadrés.
    assert group.x <= faces[0].x and group.x + group.w >= faces[1].x + faces[1].w
    assert _inside(group, 2400, 2400)


def test_all_faces_still_clamped_to_image() -> None:
    # Union trop large pour l'image : les bornes priment sur le cadrage.
    faces = [
        FaceBox(x=200, y=400, w=200, h=200, score=0.9),
        FaceBox(x=1400, y=420, w=180, h=180, score=0.8),
    ]
    group = plan(2000, 1200, faces, 400, 400, all_faces=True)
    assert _inside(group, 2000, 1200)
    assert group.w == group.h == 1200


def test_faces_sorted_largest_first_is_the_subject() -> None:
    big = FaceBox(x=800, y=800, w=300, h=300, score=0.9)
    small = FaceBox(x=1900, y=200, w=60, h=60, score=0.9)
    # Ordre volontairement inversé : `plan` ne doit pas dépendre de l'ordre reçu.
    crop = plan(2400, 2400, [small, big], 400, 400)
    assert abs((crop.x + crop.w / 2) - big.center_x) <= 1
    assert _inside(crop, 2400, 2400)


def test_openinary_transformation_string() -> None:
    crop = plan(1000, 1000, [], 400, 400)
    assert crop.to_openinary() == f"c_crop,x_{crop.x},y_{crop.y},w_{crop.w},h_{crop.h}"
