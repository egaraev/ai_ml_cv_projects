import pytest
from word_freq import count_words

@pytest.mark.parametrize(
    ("input_text", "expected"),
    [
        ('hello world', {'hello': 1, 'world': 1}),
        ('Hello, world!', {'hello': 1, 'world': 1}),
        ('hello,world', {'hello': 1, 'world': 1}),
        ('well-known', {'well': 1, 'known': 1}),
        (None, {}),
        ('', {}),
        ('hello hello world world', {'hello': 2, 'world': 2}),
        ('a1b2c3', {'a': 1, 'b': 1, 'c': 1}),
        ('   spaced   out   ', {'spaced': 1, 'out': 1}),
        (' punctuation!!!   mixed???   words...', {'punctuation': 1, 'mixed': 1, 'words': 1}),
    ]
)
def test_count_words_concrete(input_text, expected):
    """Test count_words with concrete input-output pairs from the specification."""
    assert count_words(input_text) == expected

def test_count_words_return_type():
    """Test that count_words always returns a dictionary."""
    assert isinstance(count_words("hello world"), dict)

def test_count_words_keys_are_strings():
    """Test that all keys in the returned dictionary are strings."""
    result = count_words("hello world")
    assert all(isinstance(key, str) for key in result.keys())

def test_count_words_values_are_integers():
    """Test that all values in the returned dictionary are integers."""
    result = count_words("hello world")
    assert all(isinstance(value, int) for value in result.values())

def test_count_words_values_are_positive():
    """Test that all values in the returned dictionary are positive integers."""
    result = count_words("hello world")
    assert all(value > 0 for value in result.values())

def test_count_words_no_empty_keys():
    """Test that no keys in the returned dictionary are empty strings."""
    result = count_words("hello world")
    assert all(key != "" for key in result.keys())

def test_count_words_no_invalid_characters_in_keys():
    """Test that keys contain only alphabetic characters."""
    result = count_words("hello world")
    assert all(key.isalpha() for key in result.keys())