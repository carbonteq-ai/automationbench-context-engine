"""Locations of the read-only sources this engine consumes.

Everything is overridable through environment variables so the engine can run against another
checkout. Nothing here writes to those locations.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

RESEARCH = REPO.parent

ENVIRONMENT = Path(
    os.environ.get(
        "ABCE_ENVIRONMENT", RESEARCH / "verifiers-environments/environments/automationbench_v1"
    )
)

CONTRACTS = ENVIRONMENT / "src/automationbench_v1/contracts"

CATALOG = CONTRACTS / "catalog.json"

DATA = REPO / "data"

EXCLUDED_NAMES = (
    DATA / "automationbench_heldout62_v1.json"
)  # names only; used as an exclusion filter

CACHE = Path(os.environ.get("ABCE_CACHE", REPO / ".cache"))
