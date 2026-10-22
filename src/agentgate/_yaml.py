"""Strict YAML parsing shared by the fixture and policy loaders."""

from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Hashable, Any]:
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                break  # SafeConstructor reports unhashable keys itself.
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def parse_yaml(text: str, path: Path, error: type[Exception]) -> Any:
    """Parse text as YAML, raising `error` with the file path on any problem."""
    try:
        # _UniqueKeyLoader is a SafeLoader, so tags cannot build arbitrary Python objects.
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise error(f"{path}: invalid YAML: {exc}") from exc
