"""Voyage AI embedding wrapper for the sqlite_vec memory plugin.

Spec: docs/superpowers/specs/2026-05-02-hermes-memory-design.md §1.4 (locked
decision) and §4 (read path) — voyage-3.5-lite, 512 dim, int8.

Returns each embedding as a 512-byte BLOB ready to insert into
``semantic_facts.embedding``. The store-side trigger wraps the BLOB with
``vec_int8()`` when copying it into the ``vec_facts`` virtual table.

Public API:

    await voyage_embed(["text 1", "text 2"]) -> [b"...512 bytes...", b"..."]
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional, Sequence

import httpx

logger = logging.getLogger(__name__)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MODEL = "voyage-3.5-lite"
VOYAGE_BATCH = 128  # Voyage API per-call ceiling
VOYAGE_DIM = 512
VOYAGE_DTYPE = "int8"
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3


class VoyageError(RuntimeError):
    """Raised when Voyage API repeatedly fails."""


def _api_key() -> str:
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise VoyageError(
            "VOYAGE_API_KEY is not set. Add it to ~/.hermes/.env and "
            "expose it to the hermes container via docker-compose."
        )
    return key


def _to_int8_blob(values: Sequence[int]) -> bytes:
    """Pack a list of int8 values (-128..127) into a raw 512-byte BLOB."""
    if len(values) != VOYAGE_DIM:
        raise VoyageError(
            f"Voyage returned {len(values)}-dim vector, expected {VOYAGE_DIM}"
        )
    return bytes((v + 256) & 0xFF for v in values)  # signed -> unsigned byte


async def _post_batch(
    client: httpx.AsyncClient,
    texts: List[str],
    api_key: str,
) -> List[bytes]:
    payload = {
        "model": VOYAGE_MODEL,
        "input": texts,
        "output_dtype": VOYAGE_DTYPE,
        "output_dimension": VOYAGE_DIM,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = await client.post(
                VOYAGE_URL, headers=headers, json=payload, timeout=DEFAULT_TIMEOUT
            )
        except httpx.RequestError as exc:
            if attempt == MAX_RETRIES:
                raise VoyageError(f"network error: {exc}") from exc
            await asyncio.sleep(2 ** (attempt - 1))
            continue

        if 500 <= r.status_code < 600:
            if attempt == MAX_RETRIES:
                raise VoyageError(f"Voyage 5xx: {r.status_code} {r.text[:200]}")
            await asyncio.sleep(2 ** (attempt - 1))
            continue

        if r.status_code >= 400:
            raise VoyageError(f"Voyage {r.status_code}: {r.text[:200]}")

        body = r.json()
        items = body.get("data", [])
        if len(items) != len(texts):
            raise VoyageError(
                f"Voyage returned {len(items)} items for {len(texts)} inputs"
            )
        # Voyage returns embeddings in input order (per docs/index field).
        items.sort(key=lambda d: d.get("index", 0))
        return [_to_int8_blob(d["embedding"]) for d in items]

    raise VoyageError("retry loop exhausted unexpectedly")


async def voyage_embed(
    texts: List[str],
    *,
    dim: int = VOYAGE_DIM,
    dtype: str = VOYAGE_DTYPE,
    client: Optional[httpx.AsyncClient] = None,
) -> List[bytes]:
    """Embed `texts` and return one int8 BLOB per input.

    Batches automatically at Voyage's 128-input ceiling. Retries 3x with
    exponential backoff on 5xx and network errors. Raises VoyageError on
    auth failure, 4xx, or repeated 5xx.

    `dim` and `dtype` are accepted for API symmetry but locked to the spec
    values; passing different values raises immediately so config drift
    fails loudly instead of silently corrupting embeddings.
    """
    if dim != VOYAGE_DIM or dtype != VOYAGE_DTYPE:
        raise VoyageError(
            f"dim/dtype locked to {VOYAGE_DIM}/{VOYAGE_DTYPE} per spec §1.4"
        )
    if not texts:
        return []

    api_key = _api_key()
    owns_client = client is None
    client = client or httpx.AsyncClient()
    try:
        out: List[bytes] = []
        for i in range(0, len(texts), VOYAGE_BATCH):
            batch = texts[i : i + VOYAGE_BATCH]
            out.extend(await _post_batch(client, batch, api_key))
        return out
    finally:
        if owns_client:
            await client.aclose()
