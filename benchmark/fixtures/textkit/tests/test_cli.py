from textkit.cli import main


def test_slug_command(capsys):
    assert main(["slug", "Hello World"]) == 0
    assert capsys.readouterr().out.strip() == "hello-world"


def test_case_command(capsys):
    assert main(["case", "snake", "UserAccountID"]) == 0
    assert capsys.readouterr().out.strip() == "user_account_id"


def test_wrap_command(capsys):
    assert main(["wrap", "--width", "9", "alpha beta gamma"]) == 0
    assert capsys.readouterr().out.splitlines() == ["alpha", "beta", "gamma"]


def test_stats_command(tmp_path, capsys):
    path = tmp_path / "essay.txt"
    path.write_text("the cat and the hat", encoding="utf-8")
    assert main(["stats", str(path), "--top", "1"]) == 0
    out = capsys.readouterr().out
    assert "words: 5" in out
    assert "the" in out.splitlines()[-1]


def test_render_command(capsys):
    assert main(["render", "Dear {name},", "name=Ada"]) == 0
    assert capsys.readouterr().out.strip() == "Dear Ada,"


def test_render_command_bad_assignment(capsys):
    assert main(["render", "x", "oops"]) == 2
    assert "key=value" in capsys.readouterr().err
