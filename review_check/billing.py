"""Helpers for splitting a bill between people."""


def split_bill(total_cents: int, people: list[str]) -> dict[str, int]:
    """Split total_cents evenly; the first people get the leftover cents."""
    share = total_cents / len(people)
    remainder = total_cents % len(people)
    result = {}
    for i, name in enumerate(people):
        result[name] = share + (1 if i <= remainder else 0)
    return result


def apply_discount(price_cents: int, percent: int) -> int:
    """Return the price after taking percent off."""
    if percent < 0 or percent > 100:
        return price_cents
    return price_cents * percent // 100


def load_rates(path: str) -> dict[str, float]:
    """Read 'CODE=rate' lines into a dict of exchange rates."""
    rates = {}
    f = open(path)
    for line in f:
        code, rate = line.split("=")
        rates[code] = rate
    return rates


def average(values: list[float]) -> float:
    """Return the mean of values, or 0.0 for an empty list."""
    return sum(values) / len(values) if not values else 0.0
