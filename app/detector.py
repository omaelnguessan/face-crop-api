from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class FaceBox:
    """Boîte englobante d'un visage, exprimée dans le repère de l'image source."""

    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2.0


class YuNetDetector:
    """Wrapper autour de `cv2.FaceDetectorYN` (modèle YuNet 2023mar).

    `cv2.FaceDetectorYN` n'est pas thread-safe : on garde une instance par
    thread via `threading.local()`, ce qui permet de détecter en parallèle
    depuis un ThreadPoolExecutor.
    """

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        top_k: int = 500,
        max_side: int = 1024,
    ) -> None:
        self._model_path = str(model_path)
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)
        self._max_side = int(max_side)
        self._local = threading.local()

        if not Path(self._model_path).is_file():
            raise FileNotFoundError(
                f"Modèle YuNet introuvable : {self._model_path}. "
                "Lance `bash scripts/download_model.sh`."
            )

    def _instance(self) -> cv2.FaceDetectorYN:
        """Retourne le détecteur propre au thread courant, en le créant au besoin."""
        detector = getattr(self._local, "detector", None)
        if detector is None:
            detector = cv2.FaceDetectorYN.create(
                model=self._model_path,
                config="",
                input_size=(320, 320),
                score_threshold=self._score_threshold,
                nms_threshold=self._nms_threshold,
                top_k=self._top_k,
            )
            self._local.detector = detector
        return detector

    def detect(self, image: np.ndarray) -> list[FaceBox]:
        """Détecte les visages et retourne les boîtes triées par aire décroissante.

        L'image est réduite si son plus grand côté dépasse `max_side`, puis les
        coordonnées sont remises à l'échelle de l'image d'origine.
        """
        if image is None or image.size == 0:
            return []

        src_h, src_w = image.shape[:2]
        if src_h < 1 or src_w < 1:
            return []

        scale = 1.0
        work = image
        longest = max(src_w, src_h)
        if longest > self._max_side:
            scale = self._max_side / float(longest)
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            work = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        work_h, work_w = work.shape[:2]
        detector = self._instance()
        detector.setInputSize((work_w, work_h))
        _, raw = detector.detect(work)

        if raw is None:
            return []

        inv = 1.0 / scale if scale != 1.0 else 1.0
        faces: list[FaceBox] = []
        for row in raw:
            x, y, w, h = (float(v) * inv for v in row[:4])
            score = float(row[-1])
            # Clamp dans les bornes de l'image source.
            x0 = max(0, min(src_w - 1, int(round(x))))
            y0 = max(0, min(src_h - 1, int(round(y))))
            w0 = max(1, min(src_w - x0, int(round(w))))
            h0 = max(1, min(src_h - y0, int(round(h))))
            faces.append(FaceBox(x=x0, y=y0, w=w0, h=h0, score=score))

        faces.sort(key=lambda f: f.area, reverse=True)
        return faces
