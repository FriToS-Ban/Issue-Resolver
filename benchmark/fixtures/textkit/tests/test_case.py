import pytest

from textkit.case import camel_case, kebab_case, pascal_case, snake_case, split_words, title_case


def test_split_words():
    assert split_words("parseHTTPResponse_v2") == ["parse", "HTTP", "Response", "v", "2"]


@pytest.mark.parametrize(
    "func, expected",
    [
        (snake_case, "user_account_id"),
        (kebab_case, "user-account-id"),
        (camel_case, "userAccountId"),
        (pascal_case, "UserAccountId"),
    ],
)
def test_identifier_cases(func, expected):
    assert func("UserAccountID") == expected


def test_snake_case_from_words():
    assert snake_case("Total amount due") == "total_amount_due"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("war and peace", "War and Peace"),
        ("gone with the wind", "Gone With the Wind"),
        ("songs for the road", "Songs for the Road"),
        ("NASA launches new probe", "NASA Launches New Probe"),
        ("what are you looking at", "What Are You Looking At"),
    ],
)
def test_title_case(text, expected):
    assert title_case(text) == expected


def test_title_case_custom_small_words():
    assert title_case("tom and jerry", small_words=[]) == "Tom And Jerry"
