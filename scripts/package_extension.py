"""Build the Chrome Web Store release ZIP for the PhishNet extension.

Validates the manifest (MV3, description length, least-privilege
permissions), strips the dev-only ``"key"``, bakes the production backend
URL into ``constants.js``, renders the store graphic assets, and emits a
versioned ZIP plus its SHA-256::

    uv run --with pillow python scripts/package_extension.py

Pillow is intentionally NOT a locked dependency (packaging-only); the
``--with`` flag fetches it ephemerally. Spec: docs/extension-packaging-deployment.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "extension"
DIST_DIR = ROOT / "dist"
STORE_ASSETS_DIR = DIST_DIR / "store_assets"

PROD_BACKEND = "https://phishnet-serving-683912591639.us-central1.run.app"

# Permissions the background/options code actually calls (least privilege).
USED_PERMISSIONS = {"tabs", "notifications", "storage"}
# Requested-but-unreferenced permissions are a CWS review flag.
BANNED_PERMISSIONS = {"alarms"}

STORE_ICON_SIZE = (128, 128)
PROMO_TILE_SIZE = (440, 280)
SCREENSHOT_SIZE = (1280, 800)
SCREENSHOT_SOURCES = ("tier1-phishing.png", "tier1-benign.png", "tier2-benign.png")


def fail(message: str) -> None:
    print(f"package_extension: ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_manifest() -> dict:
    manifest_path = EXT_DIR / "manifest.json"
    if not manifest_path.is_file():
        fail(f"{manifest_path} not found (run from the repo root)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != 3:
        fail("manifest_version must be 3")
    description = str(manifest.get("description", ""))
    if len(description) >= 132:
        fail(f"description is {len(description)} chars (CWS limit < 132)")
    permissions = set(manifest.get("permissions", []))
    banned = permissions & BANNED_PERMISSIONS
    if banned:
        fail(f"unreferenced permissions present: {sorted(banned)}")
    for name in ("background.js", "options.html", "options.js", "constants.js"):
        if not (EXT_DIR / name).is_file():
            fail(f"extension/{name} missing")
    for icon in ("icon16.png", "icon48.png", "icon128.png"):
        if not (EXT_DIR / "icons" / icon).is_file():
            fail(f"extension/icons/{icon} missing")
    return manifest


def stage_distribution(manifest: dict, stage: Path) -> str:
    """Copy the extension tree minus the dev key, backend baked to prod."""
    shutil.copytree(EXT_DIR, stage, ignore=shutil.ignore_patterns("*.map"))
    staged_manifest = stage / "manifest.json"
    data = json.loads(staged_manifest.read_text(encoding="utf-8"))
    data.pop("key", None)
    staged_manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    constants = stage / "constants.js"
    text = constants.read_text(encoding="utf-8")
    pattern = r'const PHISHNET_DEFAULT_BACKEND\s*=\s*"[^"]*";'
    replacement = f'const PHISHNET_DEFAULT_BACKEND = "{PROD_BACKEND}";'
    text, count = re.subn(pattern, replacement, text)
    if count != 1:
        fail("constants.js: PHISHNET_DEFAULT_BACKEND assignment not found")
    constants.write_text(text, encoding="utf-8")
    return str(manifest.get("version", "0.0"))


def render_assets(stage: Path) -> None:
    """Render store graphics (requires Pillow, see module docstring)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        fail("Pillow is required (uv run --with pillow ...)")
    STORE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    icon = Image.open(stage / "icons" / "icon128.png").convert("RGBA")
    icon.resize(STORE_ICON_SIZE, Image.LANCZOS).save(
        STORE_ASSETS_DIR / "store_icon_128.png"
    )

    tile = Image.new("RGBA", PROMO_TILE_SIZE, (13, 17, 23, 255))
    mark = icon.resize((160, 160), Image.LANCZOS)
    tile.alpha_composite(mark, ((PROMO_TILE_SIZE[0] - 160) // 2, 20))
    draw = ImageDraw.Draw(tile)
    label = "PhishNet Detector"
    box = draw.textbbox((0, 0), label)
    draw.text(
        ((PROMO_TILE_SIZE[0] - (box[2] - box[0])) / 2, 200),
        label,
        fill=(240, 246, 252, 255),
    )
    tile.convert("RGB").save(STORE_ASSETS_DIR / "promo_tile_440x280.png")

    for i, name in enumerate(SCREENSHOT_SOURCES, start=1):
        source = ROOT / "img" / name
        if not source.is_file():
            fail(f"img/{name} missing (screenshot source)")
        shot = Image.open(source).convert("RGB")
        shot.thumbnail(SCREENSHOT_SIZE, Image.LANCZOS)
        canvas = Image.new("RGB", SCREENSHOT_SIZE, (13, 17, 23))
        canvas.paste(
            shot,
            (
                (SCREENSHOT_SIZE[0] - shot.width) // 2,
                (SCREENSHOT_SIZE[1] - shot.height) // 2,
            ),
        )
        canvas.save(STORE_ASSETS_DIR / f"screenshot{i}.png")


def pack_zip(stage: Path, version: str) -> Path:
    """Zip the staged tree (sorted entries, no dev files)."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out = DIST_DIR / f"phishnet-extension-v{version}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage))
    return out


def main() -> None:
    manifest = load_manifest()
    with tempfile.TemporaryDirectory(prefix="phishnet-ext-") as tmp:
        stage = Path(tmp) / "extension"
        version = stage_distribution(manifest, stage)
        render_assets(stage)
        out = pack_zip(stage, version)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"package: {out} ({out.stat().st_size} bytes)")
    print(f"sha256: {digest}")
    print(f"assets: {STORE_ASSETS_DIR}")
    print("preflight: manifest MV3 ok, key stripped, backend baked, icons ok")


if __name__ == "__main__":
    main()
