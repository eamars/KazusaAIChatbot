"""Small self-contained coding fixture for native DSH workspace editing."""

from __future__ import annotations


def total_with_tax(price: float, tax_rate: float) -> float:
    """Return a price with its tax applied."""

    return round(price * tax_rate, 2)
