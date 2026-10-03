#!/usr/bin/env python3
"""
Terrarium elevation tile scraper for the Procedural Terrain Morphing VAE.

Downloads Mapzen Terrarium PNG tiles over mountainous regions, decodes them
to real elevation in metres (R*256 + G + B/256 - 32768), drops flat / ocean
tiles, slices each 256x256 tile into 64x64 patches, and saves them as a
float32 .npy array (in metres, NOT normalized - the notebook normalizes).

Attribution: elevation data via Mapzen Terrarium tiles
(https://s3.amazonaws.com/elevation-tiles-prod/terrarium/), sourced from
NASA SRTM / other open DEMs. Check the tile licence for your use case.

Usage:
    pip install requests numpy pillow
    python tile_scraper.py --regions himalayas andes alps --zoom 12 \
        --target-patches 4000 --out ./data
"""

import argparse
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import requests
from PIL import Image

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# (lat_min, lon_min, lat_max, lon_max) - mountainous regions with real relief
REGIONS = {
    "himalayas":     (27.0, 78.0, 32.0, 95.0),
    "andes":         (-33.0, -73.0, -8.0, -63.0),
    "alps":          (45.4, 5.0, 48.2, 12.0),
    "rockies":       (36.5, -114.0, 46.5, -103.0),
    "western_ghats": (8.0, 73.5, 16.5, 78.5),
}

_thread_local = threading.local()


def get_session():
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "terrain-vae-scraper/1.0 (student research project)"})
        _thread_local.session = s
    return s


def deg2frac(lat, lon, z):
    """Lat/lon -> fractional slippy-map tile coordinates."""
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat = max(min(lat, 85.0511), -85.0511)
    lat_rad = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tiles_for_bbox(lat_min, lon_min, lat_max, lon_max, z):
    x0, _ = deg2frac(lat_max, lon_min, z)
    x1, _ = deg2frac(lat_max, lon_max, z)
    _, y0 = deg2frac(lat_max, lon_min, z)  # north edge -> smaller y
    _, y1 = deg2frac(lat_min, lon_min, z)  # south edge -> larger y
    tiles = []
    for x in range(math.floor(x0), math.floor(x1) + 1):
        for y in range(math.floor(y0), math.floor(y1) + 1):
            tiles.append((z, x % (2 ** z), y))
    return tiles


def fetch_tile(z, x, y, retries=3):
    url = TILE_URL.format(z=z, x=x, y=y)
    for attempt in range(retries):
        try:
            r = get_session().get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 100:
                return Image.open(BytesIO(r.content)).convert("RGB")
            return None  # 404 / empty tile: nothing to retry
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def terrarium_to_metres(img):
    """Decode Terrarium RGB -> elevation in metres."""
    a = np.asarray(img, dtype=np.float32)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def process_tile(tile, args):
    """Download one tile -> list of kept 64x64 elevation patches (metres)."""
    z, x, y = tile
    img = fetch_tile(z, x, y)
    if img is None:
        return [], "failed"
    elev = terrarium_to_metres(img)
    if elev.std() < args.min_std:
        return [], "flat"  # ocean / plains: no useful relief
    ps = args.patch_size
    kept = []
    for i in range(0, 256, ps):
        for j in range(0, 256, ps):
            p = elev[i:i + ps, j:j + ps]
            if p.shape == (ps, ps) and (p.max() - p.min()) >= args.min_patch_range:
                kept.append(p.astype(np.float32))
    return kept, "kept"


def save_preview(patches, path, n=64):
    """Grayscale contact sheet of the first n patches (per-patch normalized)."""
    sel = patches[:n]
    grid = int(math.ceil(math.sqrt(len(sel))))
    ps = sel.shape[1]
    canvas = np.zeros((grid * ps, grid * ps), dtype=np.uint8)
    for k, p in enumerate(sel):
        r = p.max() - p.min()
        norm = (p - p.min()) / (r if r > 1e-6 else 1.0)
        canvas[(k // grid) * ps:(k // grid + 1) * ps,
               (k % grid) * ps:(k % grid + 1) * ps] = (norm * 255).astype(np.uint8)
    Image.fromarray(canvas).save(path)


def main():
    ap = argparse.ArgumentParser(description="Scrape Terrarium elevation tiles into VAE training patches.")
    ap.add_argument("--regions", nargs="+", default=["himalayas", "andes", "alps"],
                    choices=list(REGIONS), help="Mountain regions to sample.")
    ap.add_argument("--zoom", type=int, default=12, help="Slippy-map zoom (12 ~ 38 m/px).")
    ap.add_argument("--target-patches", type=int, default=4000, help="Stop after this many patches.")
    ap.add_argument("--max-tiles", type=int, default=1500, help="Max tiles to consider (randomly sampled).")
    ap.add_argument("--patch-size", type=int, default=64)
    ap.add_argument("--min-std", type=float, default=40.0, help="Min tile elevation std-dev (m) to keep.")
    ap.add_argument("--min-patch-range", type=float, default=25.0, help="Min patch elevation range (m) to keep.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="./data", help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    tiles = []
    for name in args.regions:
        tiles += tiles_for_bbox(*REGIONS[name], args.zoom)
    print(f"{len(tiles)} tiles cover the requested regions at z={args.zoom}")
    rng.shuffle(tiles)
    tiles = tiles[:args.max_tiles]
    print(f"Sampling {len(tiles)} tiles...")

    patches, stats = [], {"downloaded": 0, "flat": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_iter = ex.map(lambda t: process_tile(t, args), tiles)
        try:
            for kept, status in future_iter:
                stats["downloaded"] += 1
                if status in stats:
                    stats[status] += 1
                patches.extend(kept)
                if len(patches) >= args.target_patches:
                    break
                time.sleep(0.02)  # be polite to the tile server
        finally:
            ex.shutdown(cancel_futures=True)

    patches = np.stack(patches[:args.target_patches]).astype(np.float32)
    npy_path = os.path.join(args.out, "patches.npy")
    np.save(npy_path, patches)
    save_preview(patches, os.path.join(args.out, "preview.png"))

    meta = {
        "regions": args.regions, "zoom": args.zoom, "patch_size": args.patch_size,
        "n_patches": int(len(patches)),
        "elevation_metres": {"min": float(patches.min()), "max": float(patches.max()),
                             "mean": float(patches.mean())},
        "tile_stats": stats,
        "attribution": "Mapzen Terrarium tiles (elevation-tiles-prod), RGB decoded to metres",
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {len(patches)} patches -> {npy_path} {patches.shape} {patches.dtype}")
    print(f"Elevation range: {patches.min():.0f} m to {patches.max():.0f} m")
    print(f"Tiles: {stats}")
    print("Preview contact sheet: preview.png - eyeball it before training.")


if __name__ == "__main__":
    main()
