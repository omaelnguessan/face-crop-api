# face-crop-api

Microservice qui **calcule les coordonnées d'un crop centré sur le visage**, pour les injecter dans
une URL [Openinary](https://openinary.icoop.live) — qui supporte `c_crop,x_,y_,w_,h_` mais **pas**
la gravité `g_face` de Cloudinary.

Le service ne stocke aucune image, n'écrit aucun fichier, et ne persiste rien d'autre qu'un cache
mémoire de coordonnées.

```
Source :
https://openinary.icoop.live/t/www-idiscover-live/image/upload/v1708680528/mob_avatar/2103549847_2026-05-05_12-16-10.jpg

Résultat :
https://openinary.icoop.live/t/www-idiscover-live/image/upload/c_crop,x_312,y_98,w_540,h_540/c_fill,w_200,h_200/v1708680528/mob_avatar/2103549847_2026-05-05_12-16-10.jpg
```

Détection : **YuNet** (`face_detection_yunet_2023mar.onnx`, ~230 Ko) via `cv2.FaceDetectorYN` —
bien plus fiable que les cascades de Haar sur des photos de profil ou mal éclairées.

---

## Installation locale

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_model.sh          # récupère le .onnx dans models/
cp .env.example .env                    # optionnel : surcharge de configuration
```

### macOS Apple Silicon

Vérifie que ton Python est bien **arm64** et non x86_64 sous Rosetta — un Homebrew installé dans
`/usr/local` (au lieu de `/opt/homebrew`) fournit des binaires Intel, et la détection tourne alors
environ deux fois plus lentement (mesuré : 7,3 ms contre 3,9 ms par détection).

```bash
python3.12 -c "import platform; print(platform.machine())"   # doit afficher arm64
```

Si ça affiche `x86_64`, crée l'environnement avec un Python natif via [uv](https://docs.astral.sh/uv/) :

```bash
uv python install 3.12
uv venv --python cpython-3.12-macos-aarch64 .venv
uv pip install -r requirements.txt
```

`opencv-python-headless` et `numpy` publient des wheels arm64 : aucune compilation n'est nécessaire.

## Lancement

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive : <http://localhost:8000/docs>

### Docker

```bash
bash scripts/download_model.sh          # le compose monte ./models en volume
docker compose up --build
```

> Le volume `./models` recouvre le répertoire de l'image : le modèle doit donc être présent
> côté hôte avant `docker compose up`. Retire le volume du compose si tu préfères celui
> embarqué au build.

---

## Les trois routes

### `GET /v1/face/redirect` — **mode recommandé**

`302` vers l'URL Openinary transformée. Le CDN porte tout le trafic image ; le service ne calcule
que des coordonnées, une seule fois par image grâce au cache.

```bash
curl -i "http://localhost:8000/v1/face/redirect?url=https%3A%2F%2Fopeninary.icoop.live%2Ft%2Fwww-idiscover-live%2Fimage%2Fupload%2Fv1708680528%2Fmob_avatar%2F2103549847_2026-05-05_12-16-10.jpg&w=200&h=200"
```

### `GET /v1/face/coords` — coordonnées brutes

```bash
curl -s "http://localhost:8000/v1/face/coords?url=https%3A%2F%2Fopeninary.icoop.live%2Ft%2Fwww-idiscover-live%2Fimage%2Fupload%2Fv1708680528%2Fmob_avatar%2F2103549847_2026-05-05_12-16-10.jpg&w=200&h=200" | jq
```

```json
{
  "source": { "width": 1280, "height": 960 },
  "faces": 1,
  "crop": { "x": 312, "y": 98, "w": 540, "h": 540 },
  "transformation": "c_crop,x_312,y_98,w_540,h_540",
  "url": "https://openinary.icoop.live/t/www-idiscover-live/image/upload/c_crop,x_312,y_98,w_540,h_540/c_fill,w_200,h_200/v1708680528/mob_avatar/2103549847_2026-05-05_12-16-10.jpg"
}
```

### `GET /v1/face/render` — mode proxy

Renvoie les octets JPEG recadrés (`Cache-Control: public, max-age=2592000` + `ETag`).
À réserver aux cas où l'appelant ne peut pas suivre une redirection.

```bash
curl -s -o avatar.jpg "http://localhost:8000/v1/face/render?url=https%3A%2F%2Fopeninary.icoop.live%2Ft%2Fwww-idiscover-live%2Fimage%2Fupload%2Fv1708680528%2Fmob_avatar%2F2103549847_2026-05-05_12-16-10.jpg&w=200&h=200&quality=82"
```

### `GET /healthz`

```bash
curl -s http://localhost:8000/healthz
```

---

## Presets

Un preset fixe `w`, `h` et `zoom` en un seul paramètre, pour ne pas avoir à retenir des
valeurs de cadrage.

| Preset | `w`x`h` | `zoom` | Usage |
|---|---|---|---|
| `id` | `413x531` | `1.9` | Photo d'identité 35x45 mm à 300 dpi |

### `preset=id` — format photo d'identité

`413x531` correspond à 35x45 mm à 300 dpi, le format ICAO 9303 / ANTS (ratio 0.778).

Le cadrage est **mesuré, pas estimé** : YuNet renvoie 5 points de repère (yeux, nez, coins
de la bouche) dont `crop.measure()` déduit la ligne des yeux, le crâne, le menton et
l'inclinaison. `plan_by_head()` place ensuite la tête à **76 %** de la hauteur du cadre et
la ligne des yeux à **57,5 %** depuis le bas — les milieux des fourchettes ICAO (70–80 % et
50–65 %).

L'écart avec un cadrage déduit de la boîte englobante est net. Sur une même photo :

| Méthode | Tête | Ligne des yeux |
|---|---|---|
| `zoom=1.9` sur la boîte YuNet | 61,5 % ❌ | 45,5 % ❌ |
| `plan_by_head` sur les repères | 76,3 % ✅ | 57,7 % ✅ |

La hauteur de la boîte YuNet varie trop avec la pose et l'éclairage pour servir d'échelle.
La distance yeux→bouche, elle, est stable.

> Sans points de repère (visage trop petit, détecteur en repli), le preset retombe
> automatiquement sur le cadrage par `zoom=1.9`.

```bash
curl -s "http://localhost:8000/v1/face/coords?url=<URL>&preset=id" | jq
```

```json
{
  "source": { "width": 2000, "height": 2500 },
  "faces": 1,
  "crop": { "x": 725, "y": 597, "w": 886, "h": 1139 },
  "transformation": "c_crop,x_725,y_597,w_886,h_1139",
  "url": "https://openinary.icoop.live/.../c_crop,x_725,y_597,w_886,h_1139/c_fill,w_413,h_531/...",
  "preset": "id",
  "warnings": []
}
```

### Warnings de conformité

Sur `/v1/face/coords` avec `preset=id` uniquement, la réponse porte un tableau `warnings`.
Il est vide quand tous les contrôles passent.

| Warning | Déclencheur |
|---|---|
| `aucun visage détecté` | Repli sur un crop centré : le résultat n'est pas conforme |
| `N visages détectés` | Une photo d'identité ne doit contenir qu'un sujet |
| `confiance de détection faible` | Score YuNet du sujet < 0.90 |
| `résolution source insuffisante` | Le crop est plus petit que 413x531 et serait agrandi |
| `tête à X% du cadre, hors norme` | Hors 70–80 % : l'image est trop courte pour le cadrage visé |
| `ligne des yeux à X% du bas` | Hors 50–65 % |
| `tête inclinée de X°` | Roulis > 5°, mesuré sur la ligne des yeux |

Les trois derniers exigent les points de repère ; sans eux, un unique warning générique
signale un cadre tronqué.

Quand la tête a pu être mesurée, la réponse porte aussi un bloc `measured` :

```json
"measured": { "head_ratio": 0.763, "eye_line": 0.577, "roll_degrees": 6.7 }
```

**Ces contrôles sont indicatifs, pas une validation officielle.** Ils ne portent que sur ce
qui est mesurable depuis la détection : nombre de visages, confiance, résolution, cadrage.
La norme exige aussi un fond uni clair, une expression neutre bouche fermée, un regard vers
l'objectif et une tête droite — rien de tout cela n'est vérifié ici, `YuNet` ne renvoyant
qu'une boîte englobante et un score.

Les constantes anthropométriques de `crop.py` (`CROWN_ABOVE_EYES = 1.70`,
`CHIN_BELOW_EYES = 1.60`, exprimées en distances yeux→bouche) restent des moyennes : les
yeux tombent à peu près au milieu vertical de la tête, mais la morphologie varie. Ce qui est
réellement mesuré, c'est la ligne des yeux et l'inclinaison ; le crâne et le menton sont
extrapolés. Sur un corpus représentatif, ces deux facteurs valent d'être recalibrés.

---

## Paramètres

| Paramètre | Défaut | Routes | Description |
|---|---|---|---|
| `url` | — (requis) | toutes | URL `https` de l'image source, sur un hôte en liste blanche |
| `w` | `400` | toutes | Largeur cible (16–2000) |
| `h` | `400` | toutes | Hauteur cible (16–2000) |
| `zoom` | `2.6` | toutes | Hauteur du crop en multiples de la hauteur du visage (2.6 ≈ portrait serré, 4+ ≈ buste) |
| `all_faces` | `false` | toutes | Englober tous les visages détectés plutôt que le plus grand |
| `preset` | — | toutes | Jeu de valeurs pour `w`/`h`/`zoom` (voir [Presets](#presets)) |
| `quality` | `82` | `/render` | Qualité JPEG (1–100) |

Un paramètre passé explicitement l'emporte toujours sur la valeur du preset :
`?preset=id&w=350&h=450` garde le `zoom` du preset mais impose ses propres dimensions.

### Codes de réponse

| Code | Cause |
|---|---|
| `403` | Schéma ≠ `https`, ou hôte hors liste blanche |
| `400` | `w`/`h` hors bornes, `zoom` hors bornes, ou `preset` inconnu |
| `422` | Source non téléchargeable, trop volumineuse, non décodable, ou URL sans segment `upload` |
| `503` | Modèle ONNX absent (lance `scripts/download_model.sh`) |

---

## Configuration

Toutes surchargeables par variables d'environnement (ou `.env`) :

| Variable | Défaut | Rôle |
|---|---|---|
| `ALLOWED_HOSTS` | `openinary.icoop.live` | Liste blanche d'hôtes source, séparés par des virgules |
| `MAX_DIMENSION` | `2000` | Dimension cible maximale |
| `MAX_BYTES` | `12582912` | Plafond strict de téléchargement (12 Mo) |
| `SCORE_THRESHOLD` | `0.75` | Seuil de confiance YuNet |
| `NMS_THRESHOLD` | `0.3` | Seuil de suppression des non-maxima |
| `DEFAULT_ZOOM` | `2.6` | Zoom par défaut |
| `WORKERS` | `4` | Taille du `ThreadPoolExecutor` (décodage + détection) |
| `CACHE_TTL` | `604800` | TTL du cache de coordonnées (7 jours) |

---

## Réglages

Deux paramètres à caler en production :

**`SCORE_THRESHOLD` (défaut `0.75`)** — valeur prudente : peu de faux positifs, mais des avatars
de mauvaise qualité (flous, très compressés, de profil) peuvent être ratés et retomber sur le crop
centré. Si tu constates trop de ratés, descends progressivement vers `0.6`. En dessous, YuNet
commence à détecter des « visages » dans des textures — un crop centré sur du bruit est pire qu'un
crop centré tout court.

**`DEFAULT_ZOOM` (défaut `2.6`)** — non mesurable automatiquement : cale-le visuellement sur une
trentaine d'avatars réels et variés. `2.6` donne un portrait serré (tête + haut des épaules).
Monte vers `3.5`–`4.5` pour un cadrage buste, plus indulgent quand la boîte de détection est
imprécise. Rends-le surchargeable par appel via `?zoom=` pour distinguer, par exemple, les vignettes
de liste (serré) des en-têtes de profil (large).

Le décalage vertical (`HEAD_ROOM = 0.15` dans `app/crop.py`) remonte le cadre de 15 % de la hauteur
du visage. Sans lui, le cadrage paraît systématiquement trop bas.

---

## Sécurité

Une API qui télécharge une URL fournie par le client est une SSRF ouverte. La dépendance `guard`
(`app/main.py`), exécutée **avant tout accès réseau** :

- exige le schéma `https` ;
- parse l'URL avec `urlparse` et compare le **hostname** à la liste blanche (jamais une regex sur
  l'URL entière : `https://attacker.tld/openinary.icoop.live/…` doit échouer) → `403` ;
- valide `16 ≤ w,h ≤ 2000` → `400`.

En complément : `follow_redirects=False` (pas de rebond vers un hôte interne), timeout de 5 s, et
plafond strict de 12 Mo interrompu **pendant** le streaming, avant de tout charger en mémoire.

---

## Client PHP (avec repli)

Si le service est indisponible, on retombe sur `c_fill,g_center` : l'avatar s'affiche recadré
au centre plutôt que de casser la page.

```php
<?php
/**
 * Construit une URL Openinary recadrée sur le visage.
 * Retombe sur c_fill,g_center si le microservice ne répond pas.
 */
function avatarUrl(string $sourceUrl, int $w = 200, int $h = 200, float $zoom = 2.6): string
{
    $fallback = insertTransform($sourceUrl, "c_fill,g_center,w_{$w},h_{$h}");

    $endpoint = 'http://face-crop-api:8000/v1/face/coords?' . http_build_query([
        'url'  => $sourceUrl,
        'w'    => $w,
        'h'    => $h,
        'zoom' => $zoom,
    ]);

    $ch = curl_init($endpoint);
    curl_setopt_array($ch, [
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_CONNECTTIMEOUT => 1,   // le service est interne : soit il répond vite, soit on passe
        CURLOPT_TIMEOUT        => 2,
    ]);
    $body   = curl_exec($ch);
    $status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($body === false || $status !== 200) {
        return $fallback;
    }

    $data = json_decode($body, true);
    if (!is_array($data) || empty($data['transformation'])) {
        return $fallback;
    }

    return insertTransform($sourceUrl, $data['transformation'] . "/c_fill,w_{$w},h_{$h}");
}

/** Insère un segment de transformation juste après `/upload/`. */
function insertTransform(string $url, string $transform): string
{
    $pos = strpos($url, '/upload/');
    if ($pos === false) {
        return $url;
    }
    return substr($url, 0, $pos + 8) . $transform . '/' . substr($url, $pos + 8);
}

// Usage
$src = 'https://openinary.icoop.live/t/www-idiscover-live/image/upload/v1708680528/mob_avatar/2103549847_2026-05-05_12-16-10.jpg';
echo '<img src="' . htmlspecialchars(avatarUrl($src, 200, 200)) . '" alt="avatar">';
```

---

## Tests

```bash
pytest -q          # 32 tests, ni réseau ni modèle requis
```

`test_crop.py` teste `plan()` de façon pure (ni réseau ni modèle) : bornes de l'image, respect du
ratio, fallback centré sans visage. `test_api.py` utilise `TestClient` avec le service mocké :
liste blanche, bornes de dimensions, et construction exacte de l'URL de redirection.

### Tester la détection sur de vraies images

Les tests ne touchent volontairement ni au réseau ni au modèle. Pour vérifier que YuNet est bien
chargé et détecte, passe par le détecteur directement :

```bash
python - <<'EOF'
import cv2
from app.config import get_settings
from app.detector import YuNetDetector
from app.crop import plan

s = get_settings()
d = YuNetDetector(s.model_path, s.score_threshold, s.nms_threshold)
img = cv2.imread("mon_avatar.jpg")
h, w = img.shape[:2]
faces = d.detect(img)
print(f"{w}x{h} — {len(faces)} visage(s)", [(f.x, f.y, f.w, f.h, round(f.score, 2)) for f in faces])
crop = plan(w, h, faces, 200, 200)
print(crop.to_openinary())
cv2.imwrite("crop.jpg", img[crop.y:crop.y + crop.h, crop.x:crop.x + crop.w])
EOF
```

C'est aussi la façon la plus rapide de caler `DEFAULT_ZOOM` : boucle sur un dossier d'avatars réels
et regarde les `crop.jpg` produits.

> Un serveur d'images local en `http://` ne passera pas le garde-fou (schéma `https` exigé). Pour
> tester la chaîne complète hors Openinary, appelle `FaceService.detect()` / `.render()`
> directement, ou ajoute ton hôte de test à `ALLOWED_HOSTS` et sers-le en TLS.

---

## Architecture

```
app/
├── config.py     Settings pydantic-settings, surchargeables par env
├── detector.py   YuNetDetector — une instance par thread (FaceDetectorYN n'est pas thread-safe),
│                 réduction à 1024 px max puis remise à l'échelle ; expose la boîte,
│                 les 5 points de repère et le score
├── crop.py       plan() par zoom, plan_by_head() par géométrie mesurée, measure()
│                 depuis les points de repère — pur, testable sans dépendance
├── presets.py    presets de cadrage (`id`) et contrôles indicatifs de conformité
├── cache.py      TTLCache dict + Lock, 200 000 entrées, 7 jours ; coordonnées seulement
├── service.py    fetch streaming plafonné, décodage/détection dans un ThreadPoolExecutor
└── main.py       routes FastAPI, guard anti-SSRF, lifespan
```

Le décodage et la détection sont bloquants et gourmands en CPU : ils tournent dans un
`ThreadPoolExecutor` via `run_in_executor`, jamais dans la boucle d'événements.
