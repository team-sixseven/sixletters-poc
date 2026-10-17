"""Convenience commands remain read-only until an explicit result selection."""

import sys

import pytest

from sixletters import __main__ as cli


def test_interactive_mistake_retries_same_catalog_and_cancel_is_explicit(monkeypatch):
    answers = iter(["9", "a", "1 1", "2 1"])
    monkeypatch.setattr(cli, "prompt", lambda *_: next(answers))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert cli._choose_indexes(2) == [1, 0]
    monkeypatch.setattr(cli, "prompt", lambda *_: "q")
    with pytest.raises(cli.SelectionCancelled):
        cli._choose_indexes(2)


def test_noninteractive_invalid_selection_fails_without_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "prompt", lambda *_: calls.append(1) or "9")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError):
        cli._choose_indexes(2)
    assert calls == [1]


def test_interactive_empty_input_retries_but_eof_fails(monkeypatch):
    answers = iter(["", "1 2"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    assert cli._choose_indexes(2) == [0, 1]

    def end_input(*_):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)
    with pytest.raises(ValueError, match="No input"):
        cli._choose_indexes(2)
