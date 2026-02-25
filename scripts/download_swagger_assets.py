#!/usr/bin/env python
from __future__ import annotations

import base64
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

ASSET_URLS = {
    "swagger-ui-bundle.js": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
    "swagger-ui.css": "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    "redoc.standalone.js": "https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
}

# 1x1 transparent PNG fallback.
FAVICON_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8B9p0AAAAASUVORK5CYII="
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _target_dir() -> Path:
    return _project_root() / "app" / "static" / "swagger"


def _copy_from_fastapi_static(target_dir: Path) -> set[str]:
    copied: set[str] = set()
    try:
        import fastapi  # type: ignore
    except Exception:
        return copied

    static_dir = Path(fastapi.__file__).resolve().parent / "static"
    if not static_dir.exists():
        return copied

    for name in (*ASSET_URLS.keys(), "favicon.png"):
        source = static_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)
            copied.add(name)
    return copied


def _download_asset(name: str, url: str, target_dir: Path) -> bool:
    try:
        with urlopen(url, timeout=60) as response:
            content = response.read()
    except Exception:
        return False

    if not content:
        return False

    (target_dir / name).write_bytes(content)
    return True


def main() -> int:
    target_dir = _target_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    resolved_assets: set[str] = set()
    resolved_assets.update(_copy_from_fastapi_static(target_dir))

    for name, url in ASSET_URLS.items():
        if name in resolved_assets:
            continue
        if _download_asset(name, url, target_dir):
            resolved_assets.add(name)

    favicon_path = target_dir / "favicon.png"
    if "favicon.png" not in resolved_assets:
        if not favicon_path.exists():
            favicon_path.write_bytes(base64.b64decode(FAVICON_PNG_BASE64))
        resolved_assets.add("favicon.png")

    missing_assets = sorted(set(ASSET_URLS) - resolved_assets)
    if missing_assets:
        sys.stderr.write(
            "Failed to prepare Swagger/ReDoc assets: "
            + ", ".join(missing_assets)
            + ".\n"
        )
        return 1

    print(f"Swagger assets available in: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
