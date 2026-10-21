from importlib.metadata import version

import agentgate


def test_version_matches_installed_metadata() -> None:
    assert agentgate.__version__ == version("agentgate")
