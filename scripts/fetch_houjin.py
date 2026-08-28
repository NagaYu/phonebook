#!/usr/bin/env python3
"""Fetch data from Japan's National Tax Agency Corporate Number site.

**Terms of use (read this)**
----------------------------
Source: NTA Corporate Number Publication Site
        https://www.houjin-bangou.nta.go.jp/

- The information published there may be used freely -- reproduction, public
  transmission, translation and adaptation included -- under terms conforming to
  the Japanese Government's Public Data License (Version 1.0). Commercial use is
  permitted.
- **Crediting the source is required.** Example:
  "Source: National Tax Agency Corporate Number Publication Site (NTA) (page URL)".
- If you edit or process the data, you must **also state that you did so**, e.g.
  "Created by processing the NTA Corporate Number Publication Site data".
- Publishing or using it in a manner that suggests the government produced your
  work is prohibited.
- Always check the current terms at
  https://www.houjin-bangou.nta.go.jp/pc/riyokiyaku/. The summary above reflects
  2026-08 and is not the terms themselves.

Policy of this repository
-------------------------
- **The raw data is not redistributed.** Each user fetches it with this script.
- The bulk download involves choosing a format on the site, so this script does
  not scrape and does not bypass any consent flow. Either pass a zip you
  downloaded in a browser via ``--zip``, or pass the download URL you obtained
  via ``--url``.
- The Web-API requires an application id (``--app-id``).

CSV layout (Resource Definition v4.1): 30 columns, no header. Column 29 is the
furigana, defined as "full-width katakana and the prolonged mark only", blank
when unregistered.

Usage:
    # 1) verify and unpack a zip downloaded in a browser
    python scripts/fetch_houjin.py --zip ~/Downloads/00_zenkoku_all_20260801.zip --out data/raw --accept-terms
    # 2) download from a URL you already obtained
    python scripts/fetch_houjin.py --url "https://www.houjin-bangou.nta.go.jp/..." --out data/raw --accept-terms
    # 3) fetch a few records by corporate number through the Web-API (smoke test)
    python scripts/fetch_houjin.py --app-id XXXX --numbers 7000012050002 --out data/raw --accept-terms
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TERMS_URL = "https://www.houjin-bangou.nta.go.jp/pc/riyokiyaku/"
DOWNLOAD_URL = "https://www.houjin-bangou.nta.go.jp/download/zenken/"
WEBAPI_BASE = "https://api.houjin-bangou.nta.go.jp/4"

ATTRIBUTION = (
    "Source: National Tax Agency Corporate Number Publication Site (NTA), processed"
)


def print_terms() -> None:
    print("=" * 78)
    print("NTA Corporate Number Publication Site - terms of use (summary)")
    print("=" * 78)
    print(f"  Terms       : {TERMS_URL}")
    print(f"  Bulk data   : {DOWNLOAD_URL}")
    print("  - Conforms to the Public Data License (Version 1.0). Commercial use and")
    print("    redistribution are both permitted.")
    print("  - Crediting the source is required. If you process the data, you must also")
    print("    state that you did so.")
    print("  - Publishing it as if the government produced your work is prohibited.")
    print(f"  - Attribution used by this repository: {ATTRIBUTION}")
    print("=" * 78)


def verify_zip(path: Path, out_dir: Path) -> int:
    """Inspect the zip and unpack the CSV files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".csv", ".asc", ".txt"))]
        if not members:
            print("no CSV found inside the zip", file=sys.stderr)
            return 2
        print(f"files in zip: {members}")
        for m in members:
            target = out_dir / Path(m).name
            with zf.open(m) as src, target.open("wb") as dst:
                dst.write(src.read())
            print(f"  extracted: {target} ({target.stat().st_size/1e6:.1f} MB)")
    print("\nNote: the site offers a Shift-JIS edition and a Unicode edition.")
    print("      Pass --encoding cp932 or --encoding utf-8 to build_dataset.py accordingly.")
    return 0


def download(url: str, out_dir: Path) -> int:
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("/")[-1].split("?")[0] or "houjin_download.zip"
    target = out_dir / name
    print(f"downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = 0
        with target.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                total += len(chunk)
                print(f"\r  {total/1e6:.1f} MB", end="", flush=True)
    print()
    if zipfile.is_zipfile(target):
        return verify_zip(target, out_dir)
    print(f"saved: {target}")
    return 0


def fetch_webapi(app_id: str, numbers: list[str], out_dir: Path) -> int:
    """Fetch a few corporations by number through the Web-API (smoke test)."""
    import csv

    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "webapi_sample.csv"
    params = {"id": app_id, "number": ",".join(numbers), "type": "02", "history": "0"}
    print(f"calling Web-API: {WEBAPI_BASE}/num ({len(numbers)} numbers)")
    resp = requests.get(f"{WEBAPI_BASE}/num", params=params, timeout=60)
    resp.raise_for_status()
    text = resp.content.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # The first line is a header (record count etc.), so drop it.
    body = lines[1:] if lines and lines[0].count(",") < 5 else lines
    with target.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(body) + "\n")
    print(f"saved: {target} ({len(body)} rows)")
    print("Note: the Web-API response has a slightly different column layout than the")
    print("      bulk download. Check that it has 30 columns before feeding build_dataset.py.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", dest="zip_path", default=None, help="a zip you downloaded in a browser")
    p.add_argument("--url", default=None, help="a download URL you already obtained")
    p.add_argument("--app-id", default=None, help="Web-API application id")
    p.add_argument("--numbers", nargs="*", default=[], help="corporate numbers to fetch via the Web-API")
    p.add_argument("--out", default="data/raw")
    p.add_argument(
        "--accept-terms", action="store_true",
        help="assert that you have read the terms of use",
    )
    args = p.parse_args()

    print_terms()
    if not args.accept_terms:
        print(
            "\nRead the terms of use, then re-run with --accept-terms.\n"
            "(This flag does not accept the terms on your behalf. Read the text at the "
            "URL above.)",
            file=sys.stderr,
        )
        return 1

    out_dir = Path(args.out)
    if args.zip_path:
        return verify_zip(Path(args.zip_path), out_dir)
    if args.url:
        return download(args.url, out_dir)
    if args.app_id and args.numbers:
        return fetch_webapi(args.app_id, args.numbers, out_dir)

    print(
        "\nChoose how to fetch: --zip, --url, or --app-id together with --numbers.\n"
        f"The bulk data is available at {DOWNLOAD_URL}, where you pick a format "
        "(CSV/Shift-JIS, CSV/Unicode, XML).\n"
        "To exercise the whole pipeline without any data on hand:\n"
        "    python scripts/make_synthetic.py --out data/raw/synthetic.csv\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
