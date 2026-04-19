"""Schema contract tests.

These tests guard against silent schema drift by comparing three
independent sources of truth:

1. The **contract constants** in ``tests.support`` — the single place a
   human declares what columns the flat-column dataset must contain.
2. The **test fixture** ``build_row()`` — the synthetic rows used by all
   other tests.
3. The **production parsing code** in ``config.baseline.data`` — the
   ``DOMAIN_COLUMN_PREFIXES`` dict and label/user/item parsing logic.

If the upstream HuggingFace schema changes, a developer must update the
contract constants first.  Any mismatch between contract ↔ fixture or
contract ↔ production code will cause an immediate, descriptive failure
instead of silently passing with stale test data.
"""

from __future__ import annotations

from config.baseline.data import DOMAIN_COLUMN_PREFIXES
from taac2026.domain.config import DEFAULT_SEQUENCE_NAMES
from tests.support import (
    EXPECTED_DOMAIN_PREFIXES,
    EXPECTED_ITEM_INT_PREFIX,
    EXPECTED_SCALAR_COLUMNS,
    EXPECTED_USER_INT_PREFIX,
    build_row,
)


# -- Contract ↔ fixture alignment ------------------------------------------


def test_build_row_contains_all_expected_scalar_columns() -> None:
    row = build_row(0, 1_770_000_000, True, "u1", 101)
    missing = EXPECTED_SCALAR_COLUMNS - row.keys()
    assert not missing, f"build_row() is missing scalar columns: {missing}"


def test_build_row_contains_user_int_features() -> None:
    row = build_row(0, 1_770_000_000, True, "u1", 101)
    user_cols = [k for k in row if k.startswith(EXPECTED_USER_INT_PREFIX)]
    assert user_cols, f"build_row() has no columns with prefix {EXPECTED_USER_INT_PREFIX!r}"


def test_build_row_contains_item_int_features() -> None:
    row = build_row(0, 1_770_000_000, True, "u1", 101)
    item_cols = [k for k in row if k.startswith(EXPECTED_ITEM_INT_PREFIX)]
    assert item_cols, f"build_row() has no columns with prefix {EXPECTED_ITEM_INT_PREFIX!r}"


def test_build_row_contains_all_expected_domain_sequences() -> None:
    row = build_row(0, 1_770_000_000, True, "u1", 101)
    for domain_name, prefix in EXPECTED_DOMAIN_PREFIXES.items():
        domain_cols = [k for k in row if k.startswith(prefix)]
        assert domain_cols, (
            f"build_row() has no columns for domain {domain_name!r} "
            f"(expected prefix {prefix!r})"
        )


def test_build_row_has_no_unexpected_column_prefixes() -> None:
    """Catch stale columns that no longer match the contract."""
    row = build_row(0, 1_770_000_000, True, "u1", 101)
    known_prefixes = (
        tuple(EXPECTED_SCALAR_COLUMNS)
        + (EXPECTED_USER_INT_PREFIX, EXPECTED_ITEM_INT_PREFIX)
        + tuple(EXPECTED_DOMAIN_PREFIXES.values())
    )
    for col in row:
        assert any(col == s or col.startswith(s) for s in known_prefixes), (
            f"build_row() column {col!r} does not match any known contract prefix"
        )


# -- Contract ↔ production code alignment ----------------------------------


def test_domain_column_prefixes_match_contract() -> None:
    assert dict(DOMAIN_COLUMN_PREFIXES) == dict(EXPECTED_DOMAIN_PREFIXES), (
        "DOMAIN_COLUMN_PREFIXES in data.py diverged from the schema contract"
    )


def test_sequence_names_match_contract_domains() -> None:
    assert set(DEFAULT_SEQUENCE_NAMES) == set(EXPECTED_DOMAIN_PREFIXES.keys()), (
        "DEFAULT_SEQUENCE_NAMES in config.py diverged from the schema contract"
    )
