"""Strict top-level JSON-object iteration without retaining the whole artifact."""

import json
import math


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite JSON number")
    return result


def iter_object(path, *, chunk_chars=65536, max_value_chars=4_000_000):
    """Yield strict, bounded top-level (key, value) pairs."""
    yield from _iter_container(path, True, chunk_chars, max_value_chars)


def iter_array(path, *, chunk_chars=65536, max_value_chars=4_000_000):
    """Yield strict, bounded top-level array elements."""
    yield from _iter_container(path, False, chunk_chars, max_value_chars)


def _iter_container(path, object_mode, chunk_chars, max_value_chars):
    """Yield entries, rejecting malformed, duplicate and oversized records.

    Only complete top-level values are retained. Exhaust the iterator to verify
    the trailing document. UTF-8 decoding is handled by the text stream.
    """
    if type(chunk_chars) is not int or type(max_value_chars) is not int:
        raise ValueError("integer buffer limits required")
    if not 0 < chunk_chars <= max_value_chars:
        raise ValueError("invalid buffer limits")
    decoder = json.JSONDecoder(
        object_pairs_hook=unique_object, parse_constant=invalid_constant, parse_float=finite_float
    )
    with open(path, encoding="utf-8") as stream:
        buffer, pos, eof = "", 0, False

        def refill():
            nonlocal buffer, pos, eof
            buffer = buffer[pos:]
            pos = 0
            part = stream.read(chunk_chars)
            eof = not part
            buffer += part

        def skip_space():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n":
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                refill()

        def character(expected):
            nonlocal pos
            skip_space()
            if pos >= len(buffer) or buffer[pos] != expected:
                raise ValueError(f"expected {expected!r}")
            pos += 1

        def value():
            nonlocal pos
            skip_space()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer, pos)
                    # A numeric token can be split between chunks (e.g. 12|3).
                    if not eof and (end == len(buffer) or buffer[end] not in " \t\r\n,]}:"):
                        if len(buffer) - pos > max_value_chars:
                            raise ValueError("JSON value exceeds bounded reader limit")
                        refill()
                        continue
                    if end - pos > max_value_chars:
                        raise ValueError("JSON value exceeds bounded reader limit")
                    pos = end
                    return result
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError("incomplete or malformed JSON") from error
                    if len(buffer) - pos > max_value_chars:
                        raise ValueError("JSON value exceeds bounded reader limit") from error
                    refill()

        opening, closing = ("{", "}") if object_mode else ("[", "]")
        character(opening)
        skip_space()
        seen = set()
        if pos < len(buffer) and buffer[pos] != closing:
            while True:
                if object_mode:
                    key = value()
                    if not isinstance(key, str) or key in seen:
                        raise ValueError("top-level keys must be unique strings")
                    seen.add(key)
                    character(":")
                    yield key, value()
                else:
                    yield value()
                skip_space()
                if pos < len(buffer) and buffer[pos] == closing:
                    break
                character(",")
        character(closing)
        skip_space()
        if pos != len(buffer) or not eof:
            raise ValueError("unexpected trailing JSON content")
