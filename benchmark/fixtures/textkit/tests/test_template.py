from types import SimpleNamespace

from textkit.template import missing_keys, placeholders, render


def test_render_simple():
    assert render("Hello {name}!", {"name": "Ada"}) == "Hello Ada!"


def test_render_keyword_values_override_context():
    assert render("{a}-{b}", {"a": 1, "b": 2}, b=3) == "1-3"


def test_render_nested_mapping():
    assert render("Order {order.id} for {order.customer.name}", order={"id": 7, "customer": {"name": "Lin"}}) == (
        "Order 7 for Lin"
    )


def test_render_attribute_access():
    user = SimpleNamespace(name="Grace", email="grace@example.com")
    assert render("{user.name} <{user.email}>", user=user) == "Grace <grace@example.com>"


def test_render_leaves_non_placeholder_text_alone():
    assert render("price: 5 { not a placeholder", {}) == "price: 5 { not a placeholder"


def test_placeholders_in_order():
    assert placeholders("{greeting}, {name}! {greeting} again") == ["greeting", "name"]


def test_missing_keys():
    assert missing_keys("{a} {b.c} {d}", {"a": 1, "b": {}}) == ["b.c", "d"]
