"""Fetch Ornn spot order books and emit a SkyPilot catalog CSV.

Ornn (https://compute.ornn.com) has ONE programmatic surface: the hosted
MCP server (JSON-RPC over HTTP; OAuth credential — no REST API key exists).
The stock signal for the catalog is the **spot book's ask surface**: a book
with a non-null ``bestAskPricePerGpuHourMicroUsd`` has relisted capacity we
can rent by the GPU-hour; a null best-ask book has no supply RIGHT NOW
(single polls are never stock events — the book behind an idle relist can
vanish when a reservation starts or the seller delists).

Rows are emitted ONLY for books with a live ask (never quote free), one row
per (slug, gpu-count) pair at counts 1..8, priced **ask x count** (the
whole-box hourly the CSV Price column carries; books price PER GPU).
Instance types are stable tokens ``ornn:<gpu-slug>:<gpus>`` — order and
reservation ids are ephemeral and never catalogued (the QuantaCloud
pattern); the provisioner re-parses the token and re-resolves the live book
at launch.

Zero-row semantics (per the receipts, 2026-10-08): an all-null-ask fetch is
cross-checked against ``ornn_schedules_list`` — the Reserve/Market listing
surface — before it is written as zero-stock. Any listing with
``spotGpusOffered > 0`` for a slug with a null-ask book CONTRADICTS the
empty (a filter bug on our side, not honest zero-stock): refuse and raise.
Agreement writes the header-only catalog (the honest "no stock" state).
A missing credential refuses to emit an empty catalog at all (an
unconfigured provider must read as "not configured", never as "no
capacity").

GpuInfo is built with fetch_latitude's canonical builder (the legacy
'Gpus' wrapper — the catalog/common DeviceMemoryGiB pass reads
``row['Gpus'][0]['MemoryInfo']['SizeInMiB']``; the #39/#40 lesson).
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from sky.adaptors import ornn as ornn_api
from sky.catalog.data_fetchers import fetch_latitude

# Mirrors the other fetchers' column set (fetch_latitude/fetch_quantacloud).
CSV_COLUMNS = [
    "InstanceType",
    "AcceleratorName",
    "AcceleratorCount",
    "vCPUs",
    "MemoryGiB",
    "Price",
    "Region",
    "GpuInfo",
    "SpotPrice",
]

# The catalog's single region token: the bid API takes no region
# (``gpuSlug`` + ``gpus`` only), so Region is a namespace column.
REGION = "ornn"

# GPU slug -> (the ONE accelerator token the platform's identities use for
# that family, per-GPU VRAM in GiB). Keyed by SLUG (machine-stable, unlike
# display names). Unmapped slugs compact (strip a leading ``nvidia_``,
# uppercase, underscores removed) and pass through: an unrecognized GPU is
# a real accelerator nobody requests through our identities, and passing it
# through keeps it visible in the catalog for audit instead of silently
# hiding the book.
_ACCELERATOR_MAP: Dict[str, Tuple[str, Optional[float]]] = {
    "nvidia_rtx_pro_6000": ("RTXPRO6000", 96.0),
}

# A bid rents 1-8 GPUs on ONE node (the API's own bounds); one row per
# count at the live ask.
_MIN_GPUS_PER_BID = 1
_MAX_GPUS_PER_BID = 8


class OrnnCatalogError(RuntimeError):
    """Raised when the fetcher cannot run or the payload is unusable."""


def normalise_accelerator(gpu_slug: Optional[str]) -> str:
    slug = str(gpu_slug or "").strip().lower()
    mapped = _ACCELERATOR_MAP.get(slug)
    if mapped is not None:
        return mapped[0]
    if slug.startswith("nvidia_"):
        slug = slug[len("nvidia_"):]
    return slug.replace("_", "").upper()


def _accelerator_vram_gib(gpu_slug: str) -> Optional[float]:
    mapped = _ACCELERATOR_MAP.get(str(gpu_slug or "").strip().lower())
    return mapped[1] if mapped else None


def instance_type_token(gpu_slug: str, gpu_count: int) -> str:
    """The stable catalog identity: ``ornn:<gpu-slug>:<gpus>``.

    Order/reservation ids are ephemeral; the slug+count pair is what the
    provisioner re-resolves against the live book at launch.
    """
    if not gpu_slug or ":" in gpu_slug:
        raise OrnnCatalogError(
            f"invalid Ornn gpu slug {gpu_slug!r} (must be a catalog slug, "
            "e.g. nvidia_rtx_pro_6000, with no colons)")
    gpu_count = int(gpu_count)
    if not (_MIN_GPUS_PER_BID <= gpu_count <= _MAX_GPUS_PER_BID):
        raise OrnnCatalogError(
            f"invalid Ornn gpu count {gpu_count!r} (a bid rents "
            f"{_MIN_GPUS_PER_BID}-{_MAX_GPUS_PER_BID} GPUs on ONE node)")
    return f"ornn:{gpu_slug}:{gpu_count}"


def parse_instance_type_token(instance_type: str) -> Tuple[str, int]:
    """The inverse of ``instance_type_token``; raises on garbage.

    Splits on the LAST colon: ``ornn:nvidia_rtx_pro_6000:1`` -> the slug
    may itself contain the cloud prefix ``ornn:`` stripped only at the head,
    so the slug is whatever sits between the first and last colon.
    """
    raw = str(instance_type or "").strip()
    if not raw.lower().startswith("ornn:") or raw.count(":") < 2:
        raise OrnnCatalogError(
            f"invalid Ornn instance type {instance_type!r} (expected "
            "'ornn:<gpu-slug>:<gpus>', e.g. ornn:nvidia_rtx_pro_6000:1)")
    body = raw[len("ornn:"):]
    slug, _, count = body.rpartition(":")
    if not slug or not count.isdigit():
        raise OrnnCatalogError(
            f"invalid Ornn instance type {instance_type!r} (expected "
            "'ornn:<gpu-slug>:<gpus>', e.g. ornn:nvidia_rtx_pro_6000:1)")
    gpu_count = int(count)
    if not (_MIN_GPUS_PER_BID <= gpu_count <= _MAX_GPUS_PER_BID):
        raise OrnnCatalogError(
            f"invalid Ornn gpu count {count!r} in {instance_type!r} (a bid "
            f"rents {_MIN_GPUS_PER_BID}-{_MAX_GPUS_PER_BID} GPUs on ONE node)")
    return slug, gpu_count


def gpu_slug_from_instance_type(instance_type: str) -> str:
    slug, _ = parse_instance_type_token(instance_type)
    return slug


def gpu_count_from_instance_type(instance_type: str) -> int:
    _, count = parse_instance_type_token(instance_type)
    return count


def _micro_usd_to_usd(micro: Any) -> Optional[float]:
    if micro is None:
        return None
    try:
        return int(micro) / 1_000_000.0
    except (TypeError, ValueError):
        return None


def _book_ask_price_per_gpu(book: Dict[str, Any]) -> Optional[int]:
    """The book's best ask in integer micro-USD/GPU-hour, or None."""
    price = book.get("bestAskPricePerGpuHourMicroUsd")
    if price is None:
        return None
    try:
        value = int(price)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def iter_rows(books: Sequence[Dict[str, Any]]) -> Iterator[List[Any]]:
    """Yield one CSV row per (book with a live ask, gpu count 1..8).

    Rows are emitted ONLY from the books' own ask surface — the stock
    signal per the receipts — and priced ask x count (whole-box hourly).
    Never quote free: a book without a positive ask yields nothing.
    """
    for book in books:
        slug = str(book.get("gpuSlug") or "").strip()
        ask_per_gpu = _book_ask_price_per_gpu(book)
        if not slug or ask_per_gpu is None:
            continue
        accelerator = normalise_accelerator(slug)
        vram = _accelerator_vram_gib(slug)
        gpu_info = fetch_latitude._gpu_info_str(accelerator, 1, vram)  # pylint: disable=protected-access
        for count in range(_MIN_GPUS_PER_BID, _MAX_GPUS_PER_BID + 1):
            yield [
                instance_type_token(slug, count),
                accelerator,
                count,
                # The book surface carries no CPU/RAM per bid count; the
                # identity pair is (slug, count) and the launch resolves
                # the node. Empty strings keep the columns aligned.
                "",
                "",
                ask_per_gpu * count / 1_000_000.0,
                REGION,
                gpu_info,
                "",  # SpotPrice: no SkyPilot-spot tier (the market IS hourly)
            ]


def verify_zero_stock(rows: Sequence[List[Any]], books: Sequence[Dict[str,
                                                                      Any]],
                      schedules: Iterable[Dict[str, Any]]) -> None:
    """Cross-check a zero-ROW fetch before it is written as zero-stock.

    A zero-row fetch is honest zero-stock ONLY if every book's null ask
    AGREES with the listing surface (no schedule listing offers spot GPUs
# for a slug whose book shows no ask). A listing with ``spotGpusOffered >
# 0`` CONTRADICTS the empty book: that is a filter bug on our side (or a
    provider data fault), never an honest zero — refuse and raise so the
    controller reads "refused to write a wrong catalog", not "no stock".
    """
    if list(rows):
        return
    offered: Dict[str, int] = {}
    for listing in schedules:
        if not isinstance(listing, dict):
            continue
        slug = str(listing.get("gpuSlug") or listing.get("gpu_slug") or
                   "").strip()
        try:
            count = int(
                listing.get("spotGpusOffered") or
                listing.get("spot_gpus_offered") or 0)
        except (TypeError, ValueError):
            count = 0
        if slug and count > 0:
            offered[slug] = max(offered.get(slug, 0), count)
    # ANY offered listing contradicts the zero-row write — including a
    # slug whose book is MISSING from the books payload entirely
    # (CodeRabbit on skypilot-controller#529: filtering contradictions
    # by book membership made an empty/short books payload accept the
    # contradiction and write a false header-only catalog; a provider
    # data fault is exactly when the check must fire, not when it may
    # pass).
    contradicted = sorted(offered)
    if contradicted:
        raise OrnnCatalogError(
            f"refusing to write a zero-row Ornn catalog: no catalog rows "
            f"would be written, but ornn_schedules_list reports spot "
            f"GPUs offered for {contradicted!r} — a filter bug or "
            "provider data fault, never honest zero-stock (report, do "
            "not write)")


def fetch_books(
        client: Optional[ornn_api.OrnnClient] = None) -> List[Dict[str, Any]]:
    """The live order books (all GPU types, best ask/bid each)."""
    if client is None:
        client = ornn_api.OrnnClient(ornn_api.OrnnCredentials.from_env())
    return client.list_spot_books()


def fetch_schedules(
        client: Optional[ornn_api.OrnnClient] = None) -> List[Dict[str, Any]]:
    """The Reserve/Market listings (the zero-stock cross-check surface).

    Heavyweight by design — this is NEVER a purchase path, only the
    contradiction check that keeps an empty book from being written as
    zero-stock.
    """
    if client is None:
        client = ornn_api.OrnnClient(ornn_api.OrnnCredentials.from_env())
    return client.list_schedules()


def write_csv(rows: Iterable[List[Any]], out_path: str) -> int:
    """Write the catalog CSV atomically (tempfile + os.replace)."""
    import csv

    written = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(Path(out_path).parent),
                                    prefix=".ornn-catalog-",
                                    suffix=".csv")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for row in rows:
                writer.writerow(row)
                written += 1
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return written


def fetch_and_write(
    client: Optional[ornn_api.OrnnClient] = None
) -> Tuple[int, List[Dict[str, Any]]]:
    """Fetch books, cross-check empties, write the catalog; (rows, books).

    A missing credential raises OrnnAuthError (refuse to emit an empty
    catalog: unconfigured must never read as "no capacity").
    """
    books = fetch_books(client)
    rows = list(iter_rows(books))
    if not rows:
        schedules = fetch_schedules(client)
        verify_zero_stock(rows, books, schedules)
    return rows, books


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",
                        default=None,
                        help="Destination path (defaults to the in-tree "
                        "catalog location)")
    parser.add_argument("--from-fixture",
                        default=None,
                        help="Read the books from a JSON fixture instead of "
                        "the live MCP surface (tests)")
    args = parser.parse_args(argv)

    if args.from_fixture:
        with open(args.from_fixture, encoding="utf-8") as handle:
            books = json.load(handle)
        if isinstance(books, dict):
            books = books.get("orderBooks", [])
        rows = list(iter_rows(books))
        if not rows:
            schedules = []
            fixture_parent = Path(args.from_fixture).parent
            sched_path = fixture_parent / "schedules.json"
            if sched_path.is_file():
                with open(sched_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                schedules = (payload.get("listings", payload) if isinstance(
                    payload, dict) else payload)
            verify_zero_stock(rows, books, schedules)
    else:
        rows, books = fetch_and_write()

    out = args.output
    if not out:
        from sky.catalog import common as catalog_common
        out = str(
            catalog_common.get_catalog_path(f"{REGION}/vms.csv",
                                            create_new=True))
    written = write_csv(rows, out)
    print(f"Wrote {written} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
