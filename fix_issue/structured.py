"""
structured.py — Extract and validate structured (JSON) output from LLM responses.

Every important LLM output in the pipeline (triage, plan, critique) is parsed through a
small declarative schema so that downstream code receives typed, range-checked values and
the run record shows exactly which fields were invalid. No third-party dependency.

    schema = {
        "verdict": Field(str, enum=("APPROVED", "REJECTED"), required=True),
        "score":   Field(float, minimum=0, maximum=1, default=0.5),
        "files":   Field(list, items=str, default=list),
        "step":    Field(dict, schema={...}),
    }
    value, errors = validate(data, schema)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from fix_issue.llm_provider import LLMProvider


class StructuredOutputError(ValueError):
    """Raised when an LLM response cannot be turned into a schema-valid object."""


_MISSING = object()


@dataclass
class Field:
    type: type
    required: bool = False
    default: Any = _MISSING          # value or zero-arg callable
    enum: tuple | None = None
    minimum: float | None = None
    maximum: float | None = None
    items: Any = None                # type, or a dict schema for list[dict]
    schema: dict | None = None       # nested schema for dict fields
    max_items: int | None = None
    lower: bool = False              # normalise enum strings to lower case
    upper: bool = False

    def default_value(self) -> Any:
        if self.default is _MISSING:
            return None
        return self.default() if callable(self.default) else self.default


def extract_json(text: str) -> Any:
    """Return the first complete JSON object (or array) found in ``text``."""
    if text is None:
        raise StructuredOutputError("empty response")
    cleaned = re.sub(r"```(?:json|JSON)?", "", text).strip()
    # Drop <think>…</think> reasoning blocks some models emit.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            # Common LLM slip: trailing commas.
                            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                            try:
                                return json.loads(fixed)
                            except json.JSONDecodeError:
                                break
            start = cleaned.find(opener, start + 1)
    raise StructuredOutputError(f"no JSON object found in response: {text[:300]!r}")


def _coerce(value: Any, f: Field, path: str, errors: list[str]) -> Any:
    t = f.type
    if value is None:
        return f.default_value()
    if t is float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{path}: expected number, got {value!r}")
            return f.default_value()
        if f.minimum is not None and value < f.minimum:
            errors.append(f"{path}: {value} below minimum {f.minimum} (clamped)")
            value = float(f.minimum)
        if f.maximum is not None and value > f.maximum:
            errors.append(f"{path}: {value} above maximum {f.maximum} (clamped)")
            value = float(f.maximum)
        return value
    if t is int:
        try:
            value = int(float(value))
        except (TypeError, ValueError):
            errors.append(f"{path}: expected integer, got {value!r}")
            return f.default_value()
        if f.minimum is not None and value < f.minimum:
            errors.append(f"{path}: {value} below minimum {f.minimum} (clamped)")
            value = int(f.minimum)
        if f.maximum is not None and value > f.maximum:
            errors.append(f"{path}: {value} above maximum {f.maximum} (clamped)")
            value = int(f.maximum)
        return value
    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "yes", "1", "false", "no", "0"):
            return value.strip().lower() in ("true", "yes", "1")
        errors.append(f"{path}: expected boolean, got {value!r}")
        return f.default_value()
    if t is str:
        if not isinstance(value, (str, int, float)):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
            return f.default_value()
        value = str(value).strip()
        if f.lower:
            value = value.lower()
        if f.upper:
            value = value.upper()
        if f.enum is not None and value not in f.enum:
            errors.append(f"{path}: {value!r} not one of {list(f.enum)}")
            return f.default_value()
        return value
    if t is list:
        if isinstance(value, (str, dict)):
            value = [value]
        if not isinstance(value, list):
            errors.append(f"{path}: expected list, got {type(value).__name__}")
            return f.default_value()
        out = []
        for i, item in enumerate(value):
            if isinstance(f.items, dict):
                if not isinstance(item, dict):
                    errors.append(f"{path}[{i}]: expected object")
                    continue
                out.append(_validate_obj(item, f.items, f"{path}[{i}]", errors))
            elif f.items is not None:
                sub = _coerce(item, Field(f.items), f"{path}[{i}]", errors)
                if sub is not None and sub != "":
                    out.append(sub)
            else:
                out.append(item)
        if f.max_items is not None and len(out) > f.max_items:
            errors.append(f"{path}: {len(out)} items exceeds max {f.max_items} (truncated)")
            out = out[: f.max_items]
        return out
    if t is dict:
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {type(value).__name__}")
            return f.default_value()
        return _validate_obj(value, f.schema, path, errors) if f.schema else value
    return value


def _validate_obj(data: dict, schema: dict[str, Field], path: str, errors: list[str]) -> dict:
    out: dict[str, Any] = {}
    for key, f in schema.items():
        p = f"{path}.{key}" if path else key
        if key not in data or data[key] is None:
            if f.required:
                errors.append(f"{p}: required field missing")
            out[key] = f.default_value()
            continue
        out[key] = _coerce(data[key], f, p, errors)
        if f.required and out[key] in (None, ""):
            errors.append(f"{p}: required field invalid")
    return out


def validate(data: Any, schema: dict[str, Field]) -> tuple[dict, list[str]]:
    """Validate/coerce ``data`` against ``schema``. Returns (clean_dict, errors)."""
    errors: list[str] = []
    if not isinstance(data, dict):
        defaults = _validate_obj({}, schema, "", [])
        return defaults, [f"expected a JSON object, got {type(data).__name__}"]
    return _validate_obj(data, schema, "", errors), errors


def fatal_errors(errors: list[str]) -> list[str]:
    """Errors that make the object unusable (missing/invalid required fields)."""
    return [e for e in errors if "required field" in e or e.startswith("expected a JSON object")]


def generate_structured(
    llm: LLMProvider,
    prompt: str,
    system: str,
    schema: dict[str, Field],
    *,
    repair_attempts: int = 1,
    post_validate: Callable[[dict], list[str]] | None = None,
) -> tuple[dict, list[str], str]:
    """
    Call the LLM, parse JSON, validate. On fatal errors, re-prompt once with the exact
    validation errors (bounded). Returns (value, non-fatal warnings, raw_text).
    """
    raw = llm.generate(prompt, system=system)
    for attempt in range(repair_attempts + 1):
        problems: list[str]
        try:
            data = extract_json(raw)
            value, errors = validate(data, schema)
            extra = post_validate(value) if post_validate else []
            problems = fatal_errors(errors) + extra
            if not problems:
                return value, errors, raw
        except StructuredOutputError as exc:
            problems, errors = [str(exc)], []
        if attempt == repair_attempts:
            raise StructuredOutputError("; ".join(problems)[:1000])
        repair = (
            f"{prompt}\n\n### Your previous response was invalid\n"
            + "\n".join(f"- {p}" for p in problems[:15])
            + f"\n\nPrevious response (truncated):\n{raw[:1500]}\n\n"
            "Respond again with ONLY a corrected JSON object that satisfies the schema."
        )
        raw = llm.generate(repair, system=system)
    raise StructuredOutputError("unreachable")  # pragma: no cover
