from __future__ import annotations

import math

import pytest

from app.crop import CHIN_BELOW_EYES, CROWN_ABOVE_EYES, measure, plan_by_head
from app.detector import FaceBox, Landmarks
from app.presets import PRESETS, id_warnings


def face(eye_y: float = 400.0, d: float = 100.0, roll: float = 0.0) -> FaceBox:
    """Visage synthétique : yeux à `eye_y`, bouche `d` plus bas, roulis imposé."""
    half = 60.0
    dy = math.tan(math.radians(roll)) * half
    return FaceBox(
        x=440,
        y=330,
        w=220,
        h=260,
        score=0.95,
        landmarks=Landmarks(
            right_eye=(550 - half, eye_y - dy),
            left_eye=(550 + half, eye_y + dy),
            nose=(550, eye_y + d * 0.55),
            mouth_right=(520, eye_y + d),
            mouth_left=(580, eye_y + d),
        ),
    )


def test_measure_returns_none_without_landmarks() -> None:
    assert measure(FaceBox(x=0, y=0, w=10, h=10, score=0.9)) is None


def test_measure_places_crown_and_chin_around_eyes() -> None:
    m = measure(face(eye_y=400, d=100))
    assert m is not None
    assert m.crown_y == pytest.approx(400 - CROWN_ABOVE_EYES * 100, abs=0.5)
    assert m.chin_y == pytest.approx(400 + CHIN_BELOW_EYES * 100, abs=0.5)
    assert m.head_height == pytest.approx(330, abs=1)


def test_measure_reads_roll() -> None:
    assert measure(face(roll=8.0)).roll_degrees == pytest.approx(8.0, abs=0.2)
    assert measure(face(roll=0.0)).roll_degrees == pytest.approx(0.0, abs=0.2)


def test_plan_by_head_hits_icao_targets() -> None:
    m = measure(face(eye_y=800, d=120))
    crop = plan_by_head(2000, 2500, m, 413, 531, head_ratio=0.76, eye_line=0.575)
    assert m.head_height / crop.h == pytest.approx(0.76, abs=0.02)
    assert (crop.y + crop.h - m.eye_y) / crop.h == pytest.approx(0.575, abs=0.02)
    assert crop.w / crop.h == pytest.approx(413 / 531, abs=0.01)


def test_plan_by_head_stays_inside_image() -> None:
    # Visage collé au bord haut : le cadre ne peut pas sortir de l'image.
    m = measure(face(eye_y=60, d=100))
    crop = plan_by_head(800, 1000, m, 413, 531, head_ratio=0.76, eye_line=0.575)
    assert crop.x >= 0 and crop.y >= 0
    assert crop.x + crop.w <= 800 and crop.y + crop.h <= 1000


def test_id_preset_carries_icao_targets() -> None:
    preset = PRESETS["id"]
    assert preset.head_ratio == 0.76
    assert preset.eye_line == 0.575


def test_warnings_flag_excessive_roll() -> None:
    m = measure(face(eye_y=800, d=120, roll=9.0))
    crop = plan_by_head(2000, 2500, m, 413, 531, 0.76, 0.575)
    w = id_warnings([face(roll=9.0)], crop, 413, 531, 1.9, m)
    assert any("inclinée" in x for x in w)


def test_warnings_silent_on_compliant_head() -> None:
    # d=150 -> tête de 495 px, donc un crop au-delà de 413x531 sans agrandissement.
    m = measure(face(eye_y=1000, d=150))
    crop = plan_by_head(2000, 3000, m, 413, 531, 0.76, 0.575)
    assert id_warnings([face()], crop, 413, 531, 1.9, m) == []


def test_warnings_flag_head_ratio_out_of_range() -> None:
    """Image trop courte : le cadre est tronqué, la tête déborde de la fourchette."""
    m = measure(face(eye_y=200, d=120))
    crop = plan_by_head(400, 420, m, 413, 531, 0.76, 0.575)
    assert any("hors norme" in x for x in id_warnings([face()], crop, 413, 531, 1.9, m))
