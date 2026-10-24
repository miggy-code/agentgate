"""The one email address rule, shared by the policy engine and the email tool.

The engine decides on recipients and the email tool stores and sends them, so both read an
address through `parse_address` and use only what it returns. What the policy checked is
then exactly what lands in the outbox and what a mail adapter would put in a header.

The rule. A recipient string is one plain address when, after trimming leading and
trailing spaces (U+0020 only), all of these hold:

1. Every character is printable ASCII, U+0021 to U+007E. This refuses control characters
   anywhere (tab, CR, LF, NUL: header injection), whitespace inside, Unicode format
   characters (category Cf: bidi controls such as U+202E, zero-width characters) and every
   other non-ASCII character. The check runs on the string as given, before any case
   mapping, so a character whose lowercase or casefold is ASCII (the Kelvin sign U+212A
   lowers to "k", the long s U+017F casefolds to "s") cannot become an internal domain,
   and look-alikes from other scripts are refused rather than compared.
2. It is at most 254 characters and has exactly one "@".
3. The local part is 1 to 64 characters: dot-separated runs of letters, digits, "_", "+",
   "-" and "'", with no leading, trailing or doubled dot. That is narrower than RFC 5322
   on purpose. It admits what real mailboxes use and nothing a mail system may treat as
   routing or quoting: no quoted local part, no backslash, no "%" (the percent hack,
   "user%relay.example@host") and no "!" (bang paths).
4. The domain is two or more dot-separated labels of letters, digits and "-", each at most
   63 characters, at most 253 characters in all. No IP literal, no trailing dot.

The normalised address is that string lowercased, local part included. Local parts are
case-sensitive in RFC 5321, but the mail systems these customers use fold them, and storing
the lowercase form keeps the checked and the sent address identical. Anything else is
malformed: the engine treats it as external and on no list, and the email tool refuses to
store or send it.
"""

import re

MAX_ADDRESS_LENGTH = 254
MAX_LOCAL_LENGTH = 64
MAX_DOMAIN_LENGTH = 253

_ATOM = r"[a-z0-9_+'-]+"
_LABEL = r"[a-z0-9-]{1,63}"
_DOMAIN = re.compile(rf"{_LABEL}(?:\.{_LABEL})+")
_ADDRESS = re.compile(rf"({_ATOM}(?:\.{_ATOM})*)@({_DOMAIN.pattern})")
# Printable ASCII only, checked before lowercasing.
_PRINTABLE_ASCII = re.compile(r"[\x21-\x7e]+")


def _trimmed(raw: str) -> str | None:
    """The string without surrounding spaces, or None if any character is not allowed."""
    text = raw.strip(" ")
    if _PRINTABLE_ASCII.fullmatch(text) is None:
        return None
    return text


def parse_address(raw: str) -> tuple[str, str] | None:
    """Split a recipient into (normalised address, domain), or None if it is malformed."""
    text = _trimmed(raw)
    if text is None or len(text) > MAX_ADDRESS_LENGTH:
        return None
    match = _ADDRESS.fullmatch(text.lower())
    if match is None:
        return None
    local, domain = match.group(1), match.group(2)
    if len(local) > MAX_LOCAL_LENGTH or len(domain) > MAX_DOMAIN_LENGTH:
        return None
    return match.group(0), domain


def normalize_address(raw: str) -> str | None:
    """The normalised form of one plain address, or None if it is malformed."""
    parsed = parse_address(raw)
    return None if parsed is None else parsed[0]


def normalize_domain(raw: str) -> str | None:
    """A domain under the same rule (as in an "@domain" list entry), lowercased, or None."""
    text = _trimmed(raw)
    if text is None or len(text) > MAX_DOMAIN_LENGTH:
        return None
    domain = text.lower()
    return domain if _DOMAIN.fullmatch(domain) else None
