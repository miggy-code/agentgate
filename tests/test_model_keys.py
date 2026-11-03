"""Provider keys: loading, validation, and that a key never leaves the process in text."""

import copy
import json
import os
import pickle
from pathlib import Path

import pytest

from agentgate.model.keys import (
    CLIENT_CREDENTIAL_HEADERS,
    REDACTED,
    Credential,
    KeyConfigError,
    ProviderKeys,
    key_env_name,
    load_provider_keys,
    prepare_upstream_request,
    strip_query_credentials,
    validate_key,
    with_provider_credential,
)
from agentgate.policy.schema import ModelProvider

# Fake keys: the right shape for the checks below, accepted by no provider.
ANTHROPIC_KEY = "sk-ant-fake-provider-key-0123456789abcdef"
OPENAI_KEY = "sk-proj-fake-openai-key-0123456789abcdef"
GEMINI_KEY = "AIzaFakeGeminiProviderKey0123456789abcd"
COMPAT_KEY = "ds-fake-compatible-provider-key-0123456789"
ALL = {
    ModelProvider.ANTHROPIC: ANTHROPIC_KEY,
    ModelProvider.OPENAI: OPENAI_KEY,
    ModelProvider.GEMINI: GEMINI_KEY,
    ModelProvider.OPENAI_COMPATIBLE: COMPAT_KEY,
}


def _file(tmp_path: Path, content: object, mode: int = 0o600) -> Path:
    path = tmp_path / "keys.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    path.chmod(mode)
    return path


# Loading


def test_keys_load_from_the_environment() -> None:
    keys = load_provider_keys(
        environ={
            key_env_name(ModelProvider.ANTHROPIC): ANTHROPIC_KEY,
            key_env_name(ModelProvider.GEMINI): "",
            "ANTHROPIC_API_KEY": "not-ours",
            "OPENAI_API_KEY": "not-ours",
        }
    )
    assert keys.providers == {ModelProvider.ANTHROPIC}


def test_no_keys_is_an_error() -> None:
    with pytest.raises(KeyConfigError, match="no provider keys"):
        load_provider_keys(environ={"ANTHROPIC_API_KEY": "not-ours"})


def test_a_file_loads_every_provider(tmp_path: Path) -> None:
    path = _file(tmp_path, {p.value: key for p, key in ALL.items()})
    assert load_provider_keys(path).providers == set(ModelProvider)


def test_a_file_is_used_instead_of_the_environment(tmp_path: Path) -> None:
    path = _file(tmp_path, {"openai": OPENAI_KEY})
    keys = load_provider_keys(path, environ={key_env_name(ModelProvider.ANTHROPIC): ANTHROPIC_KEY})
    assert keys.providers == {ModelProvider.OPENAI}


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604, 0o666, 0o660])
def test_a_file_others_can_read_is_refused_without_showing_a_key(tmp_path: Path, mode: int) -> None:
    path = _file(tmp_path, {"openai": OPENAI_KEY}, mode)
    with pytest.raises(KeyConfigError, match="chmod 600") as exc:
        load_provider_keys(path)
    assert OPENAI_KEY not in str(exc.value)


def test_a_fifo_fails_fast(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    os.mkfifo(path)
    with pytest.raises(KeyConfigError, match="regular file"):
        load_provider_keys(path)


def test_a_symlink_is_refused(tmp_path: Path) -> None:
    target = _file(tmp_path, {"openai": OPENAI_KEY})
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(KeyConfigError, match="cannot open"):
        load_provider_keys(link)


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(KeyConfigError, match="cannot open"):
        load_provider_keys(tmp_path / "absent.json")


def test_an_oversized_file_is_refused(tmp_path: Path) -> None:
    path = _file(tmp_path, '{"openai": "' + "k" * 70_000 + '"}')
    with pytest.raises(KeyConfigError, match="too large"):
        load_provider_keys(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ({"mistral": "abc"}, "1 unknown provider names"),
        ({"openai": OPENAI_KEY, "mistral": "abc"}, "1 unknown provider names"),
        ({OPENAI_KEY: "openai"}, "1 unknown provider names"),
        ({OPENAI_KEY: "openai", ANTHROPIC_KEY: "anthropic"}, "2 unknown provider names"),
        (["openai"], "object"),
        ("null", "object"),
        ('{"openai": "' + OPENAI_KEY + '", "openai": "' + ANTHROPIC_KEY + '"}', "not valid JSON"),
        ('{"openai": "' + OPENAI_KEY, "not valid JSON"),
        ("not json " + OPENAI_KEY, "not valid JSON"),
        ({"openai": 12345}, "non-empty string"),
        ({"openai": ""}, "non-empty string"),
    ],
)
def test_a_bad_file_is_refused_and_the_error_holds_no_key(
    tmp_path: Path, content: object, message: str
) -> None:
    path = _file(tmp_path, content)
    with pytest.raises(KeyConfigError, match=message) as exc:
        load_provider_keys(path)
    # A parser error quotes the text it could not read; ours must neither show nor chain it.
    assert OPENAI_KEY not in str(exc.value) and ANTHROPIC_KEY not in str(exc.value)
    assert exc.value.__context__ is None and exc.value.__cause__ is None


# Validation


@pytest.mark.parametrize(
    "value",
    [
        "has space-in-it",
        " leading",
        "trailing\n",
        "tab\tseparated",
        "line\r\nbreak",
        "nul\x00byte",
        "del\x7fchar",
        "ünïcode-key",
        "zero​width",
        "k" * 513,
        "agt_looks-like-an-agentgate-token",
        "",
        None,
        12,
    ],
)
def test_an_unusable_key_is_refused_without_showing_it(value: object) -> None:
    with pytest.raises(KeyConfigError) as exc:
        validate_key(ModelProvider.OPENAI, value)
    assert "openai" in str(exc.value)
    if isinstance(value, str) and value:
        assert value not in str(exc.value)


def test_a_key_at_the_length_limit_is_accepted() -> None:
    assert validate_key(ModelProvider.OPENAI, "k" * 512) == "k" * 512


def test_a_provider_that_has_no_key_has_no_credential() -> None:
    keys = ProviderKeys({ModelProvider.OPENAI: OPENAI_KEY})
    assert keys.has(ModelProvider.OPENAI) and not keys.has(ModelProvider.GEMINI)
    with pytest.raises(KeyConfigError, match="gemini"):
        keys.credential(ModelProvider.GEMINI)


def test_an_empty_set_of_keys_is_refused() -> None:
    with pytest.raises(KeyConfigError):
        ProviderKeys({})


# A key never shows


def test_keys_print_without_their_values() -> None:
    keys = ProviderKeys(ALL)
    credential = keys.credential(ModelProvider.OPENAI)
    for shown in (repr(keys), str(keys), f"{keys}", f"{keys!r}", repr(credential), str(credential)):
        for key in ALL.values():
            assert key not in shown
    assert "openai" in repr(keys)


def test_keys_cannot_be_pickled_or_copied() -> None:
    keys = ProviderKeys(ALL)
    for item in (keys, keys.credential(ModelProvider.OPENAI)):
        with pytest.raises(TypeError):
            pickle.dumps(item)
        with pytest.raises(TypeError):
            copy.deepcopy(item)


def test_no_public_attribute_returns_a_key() -> None:
    keys = ProviderKeys(ALL)
    for name in dir(keys):
        if name.startswith("_") or not isinstance(getattr(keys, name), str | bytes | set):
            continue
        assert getattr(keys, name) not in ALL.values()
    assert keys.providers == set(ALL)


def test_redact_removes_every_key_from_text_and_bytes() -> None:
    keys = ProviderKeys(ALL)
    text = f"upstream said: bad key {OPENAI_KEY}; also Bearer {ANTHROPIC_KEY} and {GEMINI_KEY}"
    cleaned = keys.redact(text)
    assert isinstance(cleaned, str)
    assert all(key not in cleaned for key in ALL.values())
    assert REDACTED in cleaned and "Bearer " + REDACTED in cleaned
    raw = keys.redact(text.encode())
    assert isinstance(raw, bytes) and all(key.encode() not in raw for key in ALL.values())
    assert keys.appears_in(text) and not keys.appears_in("nothing here")


def test_redact_handles_a_key_that_contains_another() -> None:
    short, long = "abcdefgh", "abcdefgh-and-more-0123"
    keys = ProviderKeys({ModelProvider.OPENAI: short, ModelProvider.ANTHROPIC: long})
    assert keys.redact(f"x {long} y") == f"x {REDACTED} y"


# Credentials


@pytest.mark.parametrize(
    ("provider", "name", "value"),
    [
        (ModelProvider.ANTHROPIC, "x-api-key", ANTHROPIC_KEY),
        (ModelProvider.OPENAI, "authorization", f"Bearer {OPENAI_KEY}"),
        (ModelProvider.GEMINI, "x-goog-api-key", GEMINI_KEY),
        (ModelProvider.OPENAI_COMPATIBLE, "authorization", f"Bearer {COMPAT_KEY}"),
    ],
)
def test_each_provider_gets_its_own_credential_slot(
    provider: ModelProvider, name: str, value: str
) -> None:
    credential = ProviderKeys(ALL).credential(provider)
    assert (credential.name, credential.value) == (name, value)


def test_a_provider_credential_replaces_every_client_credential() -> None:
    headers = [
        ("Authorization", "Bearer client-token"),
        ("authorization", "Bearer second"),
        ("X-Api-Key", "client-token"),
        ("X-GOOG-API-KEY", "client-token"),
        ("api-key", "client-token"),
        ("Cookie", "session=client"),
        ("Proxy-Authorization", "Basic abc"),
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
    ]
    sent = with_provider_credential(headers, Credential("authorization", "Bearer provider-key"))
    assert sent == [
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
        ("authorization", "Bearer provider-key"),
    ]
    assert all(name.lower() not in CLIENT_CREDENTIAL_HEADERS for name, _ in sent[:-1])


def test_a_client_header_with_no_credential_is_left_alone() -> None:
    sent = with_provider_credential([("x-trace", "1")], Credential("x-api-key", "k"))
    assert sent == [("x-trace", "1"), ("x-api-key", "k")]


@pytest.mark.parametrize(
    ("path", "kept", "removed"),
    [
        ("/m:generateContent?alt=sse&key=abc&x=%20", "/m:generateContent?alt=sse&x=%20", ["abc"]),
        ("/m?key=a&key=b", "/m", ["a", "b"]),
        ("/v1/messages", "/v1/messages", []),
        ("/m?KEY=a", "/m", ["a"]),
        ("/m?%6Bey=a", "/m", ["a"]),
        ("/m?k%65y=a&x=1", "/m?x=1", ["a"]),
        ("/m?api_key=a&apikey=b&access_token=c&x=1", "/m?x=1", ["a", "b", "c"]),
        ("/m?keys=a&monkey=b", "/m?keys=a&monkey=b", []),
        ("/m?x=1&key", "/m?x=1", [""]),
        ("/m?key=a#frag", "/m#frag", ["a"]),
    ],
)
def test_query_credentials_are_stripped(path: str, kept: str, removed: list[str]) -> None:
    assert strip_query_credentials(path) == (kept, removed)


def test_an_upstream_request_carries_only_the_providers_credential() -> None:
    keys = ProviderKeys(ALL)
    path, headers = prepare_upstream_request(
        keys,
        ModelProvider.GEMINI,
        "/v1beta/models/m:generateContent?key=client-secret&alt=sse",
        [("X-Api-Key", "client-secret"), ("Authorization", "Bearer client-secret"), ("x-t", "1")],
    )
    assert path == "/v1beta/models/m:generateContent?alt=sse"
    assert headers == [("x-t", "1"), ("x-goog-api-key", GEMINI_KEY)]
