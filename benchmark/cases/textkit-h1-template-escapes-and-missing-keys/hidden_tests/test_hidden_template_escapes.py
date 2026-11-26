from types import SimpleNamespace

import pytest

from textkit import TemplateError, TextkitError, placeholders, render
from textkit.cli import main
from textkit.template import missing_keys


@pytest.mark.parametrize(
    "template, values, expected",
    [
        ("{{name}}", {"name": "x"}, "{name}"),
        ('{{"id": {id}}}', {"id": 5}, '{"id": 5}'),
        ("{{{name}}}", {"name": "Ada"}, "{Ada}"),
        (".btn {{ color: {color}; }}", {"color": "red"}, ".btn { color: red; }"),
        ("}} and {{", {}, "} and {"),
        ("Hello {name}!", {"name": "Ada"}, "Hello Ada!"),
    ],
)
def test_doubled_braces_are_literal(template, values, expected):
    assert render(template, values) == expected


def test_missing_key_raises_template_error():
    with pytest.raises(TemplateError) as excinfo:
        render("Dear {customer},", {})
    assert "customer" in str(excinfo.value)
    assert isinstance(excinfo.value, TextkitError)


def test_missing_nested_key_names_full_placeholder():
    with pytest.raises(TemplateError) as excinfo:
        render("Order {order.id} shipped", order={"status": "sent"})
    assert "order.id" in str(excinfo.value)


def test_missing_attribute_names_full_placeholder():
    with pytest.raises(TemplateError) as excinfo:
        render("{user.email}", user=SimpleNamespace(name="Grace"))
    assert "user.email" in str(excinfo.value)


def test_escaped_names_are_not_placeholders():
    assert placeholders("{{skip}} {use} {{also.skip}}") == ["use"]
    assert missing_keys("{{literal}} {needed}", {}) == ["needed"]


def test_cli_reports_missing_value_cleanly(capsys):
    assert main(["render", "Hi {name}, your code is {code}", "name=Ada"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("textkit: error:")
    assert "code" in err


def test_cli_renders_escaped_braces(capsys):
    assert main(["render", "{{{key}}}", "key=v"]) == 0
    assert capsys.readouterr().out.strip() == "{v}"
