"""Shared parsing utilities."""


def parse_num_bytes(s: str) -> int:
    """Parse a human-readable size string into bytes.

    Examples: '256M' → 268435456, '1G' → 1073741824, '512K' → 524288.
    Plain integers are treated as bytes.
    """
    s = s.strip().upper()
    multipliers = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)
