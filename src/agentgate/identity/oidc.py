"""An OpenID Connect issuer's signing keys: discovery, then its JWKS, cached.

- Discovery reads `<issuer>/.well-known/openid-configuration`. The document's `issuer` must
  equal the configured one exactly, and its `jwks_uri` must be https (http only on a loopback
  host), as the issuer itself.
- The JWKS is cached for its `Cache-Control` max-age, 10 minutes without one, and never less
  than the refetch interval (so a `max-age=0` set is not refetched on every assertion).
- A kid the cache does not hold triggers a refetch (a key rotation), at most one fetch per
  REFETCH_INTERVAL for the issuer, whatever the reason.
- Fetches use httpx with 5 s timeouts, no redirects, no proxy or netrc from the environment,
  and a size bound on the reply.
- An issuer whose keys cannot be loaded, or whose cached keys expired and cannot be fetched
  again, is unavailable: every assertion from it is refused (fail closed) until its keys load.
"""

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx

from agentgate._json import no_duplicate_keys
from agentgate.identity.config import IssuerConfig, Jwk, check_fetch_url, signing_keys

log = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 5.0
DEFAULT_CACHE_SECONDS = 600.0
# The most a JWKS is cached for, whatever its max-age says.
MAX_CACHE_SECONDS = 86_400.0
# At most one fetch per issuer in this long.
REFETCH_INTERVAL_SECONDS = 60.0
# A discovery document or JWKS is a few kilobytes.
MAX_DOCUMENT_BYTES = 256 * 1024

_MAX_AGE = re.compile(r"(?:^|[,\s])max-age\s*=\s*\"?(\d{1,10})\"?", re.IGNORECASE)


class IssuerUnavailable(Exception):
    """The issuer's keys are not loaded and cannot be fetched now."""


def fetch_client() -> httpx.Client:
    """The HTTP client key fetches use: 5 s timeouts, no redirects, nothing from the
    environment (proxies, netrc)."""
    return httpx.Client(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=False, trust_env=False)


class OidcKeys:
    """One OIDC issuer's signing keys. Thread-safe: verifications of several control requests
    share one instance."""

    def __init__(
        self,
        issuer: IssuerConfig,
        http: httpx.Client,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._issuer = issuer
        self._http = http
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._jwks_uri: str | None = None
        self._keys: Mapping[str, Jwk] | None = None
        self._expires = 0.0
        self._last_fetch: float | None = None
        # Why the last fetch failed, for `identity check-config` and the startup note.
        self.problem: str | None = None

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._keys is not None and self._monotonic() < self._expires

    def key_count(self) -> int:
        with self._lock:
            return 0 if self._keys is None else len(self._keys)

    def load(self) -> bool:
        """Fetch the keys now (at startup). Returns whether they loaded; never raises."""
        with self._lock:
            self._fetch()
            return self._keys is not None

    def key(self, kid: str) -> Jwk | None:
        """The issuer's key with this kid, or None if the issuer publishes none. Raises
        IssuerUnavailable when no current key set is loaded and none can be fetched now."""
        with self._lock:
            now = self._monotonic()
            if self._keys is None or now >= self._expires:
                # Expired keys are not used past their max-age: a provider that stops
                # answering also stops vouching.
                self._keys = None
                if self._may_fetch(now):
                    self._fetch()
                if self._keys is None:
                    raise IssuerUnavailable(self._issuer.name)
            found = self._keys.get(kid)
            if found is None and self._may_fetch(now):
                # An unknown kid may be a key the provider rotated in since the last fetch.
                previous = self._keys
                self._fetch()
                if self._keys is None:
                    # The refetch failed; the keys already held are still current.
                    self._keys = previous
                found = self._keys.get(kid)
            return found

    def _may_fetch(self, now: float) -> bool:
        return self._last_fetch is None or now - self._last_fetch >= REFETCH_INTERVAL_SECONDS

    def _fetch(self) -> None:
        """Discover (once) and fetch the JWKS. On failure the keys are None and `problem`
        says why. Holds the lock."""
        self._last_fetch = self._monotonic()
        try:
            if self._jwks_uri is None:
                self._jwks_uri = self._discover()
            body, cache_control = self._get(self._jwks_uri)
            keys = signing_keys(_json_object(body), strict=False)
        except (_FetchError, ValueError) as exc:
            self._keys = None
            self.problem = str(exc)
            log.warning("identity issuer %s: keys not loaded: %s", self._issuer.name, exc)
            return
        self._keys = keys
        self._expires = self._last_fetch + _cache_seconds(cache_control)
        self.problem = None

    def _discover(self) -> str:
        url = self._issuer.issuer.rstrip("/") + "/.well-known/openid-configuration"
        document = _json_object(self._get(url)[0])
        if document.get("issuer") != self._issuer.issuer:
            raise _FetchError("the discovery document names another issuer")
        jwks_uri = document.get("jwks_uri")
        if not isinstance(jwks_uri, str):
            raise _FetchError("the discovery document has no jwks_uri")
        try:
            check_fetch_url(jwks_uri, "jwks_uri")
        except ValueError as exc:
            raise _FetchError(str(exc)) from None
        return jwks_uri

    def _get(self, url: str) -> tuple[bytes, str | None]:
        """The reply's body (decoded, at most MAX_DOCUMENT_BYTES) and its Cache-Control."""
        try:
            with self._http.stream("GET", url, headers={"Accept": "application/json"}) as reply:
                if reply.status_code != 200:
                    raise _FetchError(f"{url} answered {reply.status_code}")
                body = bytearray()
                for chunk in reply.iter_bytes():
                    body += chunk
                    if len(body) > MAX_DOCUMENT_BYTES:
                        raise _FetchError(f"{url} answered more than {MAX_DOCUMENT_BYTES} bytes")
                cache_control = reply.headers.get("cache-control")
        except httpx.HTTPError as exc:
            raise _FetchError(f"{url} unreachable: {type(exc).__name__}") from None
        return bytes(body), cache_control


class _FetchError(Exception):
    pass


def _json_object(body: bytes) -> dict[str, Any]:
    try:
        data = json.loads(body.decode("utf-8"), object_pairs_hook=no_duplicate_keys)
    except (UnicodeDecodeError, ValueError):
        raise _FetchError("the reply is not a JSON object") from None
    if not isinstance(data, dict):
        raise _FetchError("the reply is not a JSON object")
    return data


def _cache_seconds(header: str | None) -> float:
    if header is None:
        return DEFAULT_CACHE_SECONDS
    found = _MAX_AGE.search(header)
    if found is None:
        return DEFAULT_CACHE_SECONDS
    return min(max(float(found[1]), REFETCH_INTERVAL_SECONDS), MAX_CACHE_SECONDS)
