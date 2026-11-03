"""Provider keys: loaded once, held only in the agentgate process, never shown.

A run's agent holds an agentgate token (agentgate.model.tokens) in place of a provider key.
The gateway strips every credential the client sent and adds the key of the routed provider
only. Keys come from an owner-only JSON file (provider name to key) or, without a file, from
AGENTGATE_<PROVIDER>_API_KEY variables.

A key is a secret that exists in one place, so nothing here lets it travel. Errors name the
provider and the file path, never a value, and are raised outside the `except` that caught
a parser error, because those quote the text they could not read. `ProviderKeys` and
`Credential` print without their values and cannot be pickled. `ProviderKeys.redact` removes
any key from text or bytes on their way to a log, an error or a response.
"""

import json
import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote_plus

from agentgate._json import no_duplicate_keys
from agentgate.policy.schema import ModelProvider

# Every agentgate token starts with this, so a key that does too is a token pasted by mistake.
TOKEN_PREFIX = "agt_"
KEY_ENV_PREFIX = "AGENTGATE_"
KEY_ENV_SUFFIX = "_API_KEY"
MAX_KEY_LENGTH = 512
# A keys file holds a handful of short strings; anything larger is not one.
MAX_KEYS_FILE_BYTES = 64 * 1024
REDACTED = "[redacted]"

# Headers that carry a client's credential. All of them are removed before a request is
# forwarded, whichever provider it goes to, so a client cannot pick the credential the
# upstream sees.
CLIENT_CREDENTIAL_HEADERS = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "x-goog-api-key", "api-key", "cookie"}
)
# The headers a client may present its token in, whichever SDK it uses.
PRESENTED_CREDENTIAL_HEADERS = ("authorization", "x-api-key", "x-goog-api-key", "api-key")
# Query parameters that carry a credential (Gemini takes its key as ?key=).
CLIENT_CREDENTIAL_QUERY_PARAMS = frozenset({"key", "api_key", "apikey", "access_token"})


class KeyConfigError(ValueError):
    """Provider keys are missing, malformed or stored unsafely. Messages never hold a value."""


def key_env_name(provider: ModelProvider) -> str:
    return f"{KEY_ENV_PREFIX}{provider.value.upper()}{KEY_ENV_SUFFIX}"


@dataclass(frozen=True)
class Credential:
    """The one header that authenticates a request to a provider."""

    name: str
    value: str = field(repr=False)

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("a Credential cannot be pickled")


class ProviderKeys:
    """The provider keys of this process. It has no way to read a key back as text, other
    than the Credential a request is sent with."""

    def __init__(self, keys: Mapping[ModelProvider, str]) -> None:
        if not keys:
            raise KeyConfigError("no provider keys are configured")
        self._keys = {provider: validate_key(provider, key) for provider, key in keys.items()}

    @property
    def providers(self) -> frozenset[ModelProvider]:
        return frozenset(self._keys)

    def has(self, provider: ModelProvider) -> bool:
        return provider in self._keys

    def credential(self, provider: ModelProvider) -> Credential:
        """The credential to send to provider. Raises KeyConfigError if it has no key."""
        key = self._keys.get(provider)
        if key is None:
            raise KeyConfigError(f"no key is configured for {provider.value}")
        if provider is ModelProvider.ANTHROPIC:
            return Credential("x-api-key", key)
        if provider is ModelProvider.GEMINI:
            return Credential("x-goog-api-key", key)
        return Credential("authorization", f"Bearer {key}")

    def redact(self, data: str | bytes) -> str | bytes:
        """data with every key, and the Bearer form of it, replaced by a placeholder."""
        for key in sorted(self._keys.values(), key=len, reverse=True):
            if isinstance(data, bytes):
                data = data.replace(key.encode(), REDACTED.encode())
            else:
                data = data.replace(key, REDACTED)
        return data

    def appears_in(self, data: str | bytes) -> bool:
        return self.redact(data) != data

    def __repr__(self) -> str:
        names = ", ".join(sorted(provider.value for provider in self._keys))
        return f"ProviderKeys(providers=[{names}])"

    __str__ = __repr__

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("ProviderKeys cannot be pickled")


def validate_key(provider: ModelProvider, value: object) -> str:
    """The key if it is usable as an HTTP header value, else KeyConfigError (no value shown)."""
    name = provider.value
    if not isinstance(value, str) or not value:
        raise KeyConfigError(f"{name} key must be a non-empty string")
    if len(value) > MAX_KEY_LENGTH:
        raise KeyConfigError(f"{name} key is longer than {MAX_KEY_LENGTH} characters")
    # Visible ASCII only: whitespace or a control character would split or smuggle a header.
    if not all(0x21 <= ord(char) <= 0x7E for char in value):
        raise KeyConfigError(f"{name} key must be visible ASCII with no whitespace")
    if value.startswith(TOKEN_PREFIX):
        raise KeyConfigError(f"{name} key looks like an agentgate token")
    return value


def load_provider_keys(
    path: Path | None = None, environ: Mapping[str, str] | None = None
) -> ProviderKeys:
    """Load keys from the file at path, else from the environment."""
    if path is not None:
        raw = _read_keys_file(path)
    else:
        env = os.environ if environ is None else environ
        raw = {
            provider: env[name]
            for provider in ModelProvider
            if env.get(name := key_env_name(provider))
        }
    if not raw:
        names = ", ".join(key_env_name(provider) for provider in ModelProvider)
        raise KeyConfigError(f"no provider keys: pass --keys FILE or set {names}")
    return ProviderKeys(raw)


def _read_keys_file(path: Path) -> dict[ModelProvider, str]:
    # O_NONBLOCK keeps a FIFO placed at the path from hanging startup in open(); O_NOFOLLOW
    # refuses a symlink, which could point somewhere a reader of the path never intended.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise KeyConfigError(f"cannot open keys file {path}: {exc.strerror}") from None
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise KeyConfigError(f"keys file {path} must be a regular file")
        if info.st_mode & 0o077:
            raise KeyConfigError(f"keys file {path} must not be readable by others (chmod 600)")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise KeyConfigError(f"keys file {path} must be owned by the user running agentgate")
        if info.st_size > MAX_KEYS_FILE_BYTES:
            raise KeyConfigError(f"keys file {path} is too large")
        content = handle.read(MAX_KEYS_FILE_BYTES + 1)
    return _parse_keys(content, path)


def _parse_keys(content: bytes, path: Path) -> dict[ModelProvider, str]:
    data: object = None
    unreadable = False
    try:
        data = json.loads(content, object_pairs_hook=no_duplicate_keys)
    except (ValueError, RecursionError):
        # Parser errors quote the text they could not read, which is the secret.
        unreadable = True
    if unreadable:
        raise KeyConfigError(f"keys file {path} is not valid JSON")
    if not isinstance(data, dict):
        raise KeyConfigError(f"keys file {path} must hold an object of provider to key")
    keys: dict[ModelProvider, str] = {}
    known = {p.value for p in ModelProvider}
    unknown = [name for name in data if name not in known]
    if unknown:
        # Only a count: a file with its keys and providers swapped has a key where a name
        # should be, and the name would print it.
        raise KeyConfigError(f"keys file {path} has {len(unknown)} unknown provider names")
    for provider in ModelProvider:
        if provider.value in data:
            keys[provider] = validate_key(provider, data[provider.value])
    return keys


# Client credentials


def strip_query_credentials(path: str) -> tuple[str, list[str]]:
    """path without its credential query parameters, and the values removed.

    Other parameters keep their encoding and order. Names are compared decoded and case
    folded, so `%6Bey=` and `KEY=` are removed like `key=`.
    """
    base, question, query = path.partition("?")
    if not question:
        return path, []
    query, hash_sep, fragment = query.partition("#")
    kept: list[str] = []
    removed: list[str] = []
    for part in query.split("&"):
        name, _, value = part.partition("=")
        if unquote_plus(name).casefold() in CLIENT_CREDENTIAL_QUERY_PARAMS:
            removed.append(unquote_plus(value))
        else:
            kept.append(part)
    rebuilt = base + ("?" + "&".join(kept) if kept else "")
    return rebuilt + (hash_sep + fragment if hash_sep else ""), removed


def with_provider_credential(
    headers: Iterable[tuple[str, str]], credential: Credential
) -> list[tuple[str, str]]:
    """headers with every client credential removed and the provider's added.

    Meant to run after hop-by-hop filtering, so a client `Connection` header cannot name the
    injected credential for removal.
    """
    kept = [
        (name, value) for name, value in headers if name.lower() not in CLIENT_CREDENTIAL_HEADERS
    ]
    return [*kept, (credential.name, credential.value)]


def prepare_upstream_request(
    keys: ProviderKeys,
    provider: ModelProvider,
    path: str,
    headers: Iterable[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]]]:
    """The path and headers to send to a provider: the client's credentials removed from both
    and the provider's own added. This is the one function the proxy's hook calls to build an
    upstream request, so what it returns is exactly what the provider can see of the
    client's credentials."""
    clean_path, _ = strip_query_credentials(path)
    return clean_path, with_provider_credential(headers, keys.credential(provider))
