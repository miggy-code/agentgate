"""Strict JSON parsing shared by the key and rate-table loaders."""


def no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """An object_pairs_hook that rejects a repeated key instead of keeping the last one."""
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate key")
        result[name] = value
    return result
