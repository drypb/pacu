from pacu.features.distribution_features import *
from pacu.features.distribution_features import _CHAR_INDEX, _CHAR_SPACE_LEN

def test_strip_url():
    assert strip_url("https://www.example.com") == "www.example.com"
    assert strip_url("http://example.com") == "example.com"
    assert strip_url("ftp://ftp.example.com") == "ftp.example.com"
    assert strip_url("example.com") == "example.com"
    assert strip_url("https://sub.example.com/123") == "sub.example.com/123"
    assert strip_url("http://test-site.com/path/to/page") == "test-site.com/path/to/page"
    assert strip_url("https://example.com?param=value") == "example.com?param=value"
    assert strip_url("http://example.com/search?q=test") == "example.com/search?q=test"
    assert strip_url("https://example.com:8080") == "example.com:8080"
    assert strip_url("ftp://localhost:21") == "localhost:21"
    assert strip_url("https://sub.domain.example.com/path/to/resource") == "sub.domain.example.com/path/to/resource"
    assert strip_url("http://example.com/?r=http://redirect.org") == "example.com/?r=http://redirect.org"
    assert strip_url("https:// some weird stuff here.com/ space     /") == "someweirdstuffhere.com/space/"
    assert strip_url("http://somewhere21.com/this is a tab\texample\t/") == "somewhere21.com/thisisatabexample/"
    assert strip_url("     https://leading.and.trailing.spaces.com/       ") == "leading.and.trailing.spaces.com/"
    assert strip_url("ftp:// \t\f\r\n\v  ok.com") == "ok.com"

def test_char_dist():
    # teste 1
    tmp = char_dist("http://aaaa.com")
    assert tmp[_CHAR_INDEX['a']] == (4 / 8)

    assert len(tmp) == _CHAR_SPACE_LEN
    assert abs(sum(tmp) - 1.0) < 1e-6
    for value in tmp:
        assert 0.0 <= value <= 1.0

    # teste 2
    tmp = char_dist("ftp://abcABC.com")
    assert tmp[_CHAR_INDEX['a']] == (1 / 10)
    assert tmp[_CHAR_INDEX['A']] == (1 / 10)
    assert tmp[_CHAR_INDEX['b']] == (1 / 10)
    assert tmp[_CHAR_INDEX['B']] == (1 / 10)
    assert tmp[_CHAR_INDEX['c']] == (2 / 10)
    assert tmp[_CHAR_INDEX['C']] == (1 / 10)
    assert tmp[_CHAR_INDEX['.']] == (1 / 10)
    assert tmp[_CHAR_INDEX['o']] == (1 / 10)
    assert tmp[_CHAR_INDEX['m']] == (1 / 10)

    assert len(tmp) == _CHAR_SPACE_LEN
    assert abs(sum(tmp) - 1.0) < 1e-6
    for value in tmp:
        assert 0.0 <= value <= 1.0

    # teste 3
    tmp = char_dist("http://abc-123.com")
    assert tmp[_CHAR_INDEX['x']] == 0
    assert tmp[_CHAR_INDEX['y']] == 0
    assert tmp[_CHAR_INDEX['z']] == 0
    assert tmp[_CHAR_INDEX['7']] == 0
    assert tmp[_CHAR_INDEX['_']] == 0
    assert tmp[_CHAR_INDEX['w']] == 0
    assert tmp[_CHAR_INDEX['f']] == 0

    assert len(tmp) == _CHAR_SPACE_LEN
    assert abs(sum(tmp) - 1.0) < 1e-6
    for value in tmp:
        assert 0.0 <= value <= 1.0

def test_bigram_dist():
    # teste 1
    tmp = bigram_dist("https://abc.com")
    assert tmp[_CHAR_INDEX['a'] * _CHAR_SPACE_LEN + _CHAR_INDEX['b']] == (1 / 6)
    assert tmp[_CHAR_INDEX['b'] * _CHAR_SPACE_LEN + _CHAR_INDEX['c']] == (1 / 6)
    assert tmp[_CHAR_INDEX['c'] * _CHAR_SPACE_LEN + _CHAR_INDEX['.']] == (1 / 6)
    assert tmp[_CHAR_INDEX['.'] * _CHAR_SPACE_LEN + _CHAR_INDEX['c']] == (1 / 6)
    assert tmp[_CHAR_INDEX['c'] * _CHAR_SPACE_LEN + _CHAR_INDEX['o']] == (1 / 6)
    assert tmp[_CHAR_INDEX['o'] * _CHAR_SPACE_LEN + _CHAR_INDEX['m']] == (1 / 6)
    assert tmp[_CHAR_INDEX['a'] * _CHAR_SPACE_LEN + _CHAR_INDEX['a']] == 0
    assert tmp[_CHAR_INDEX['b'] * _CHAR_SPACE_LEN + _CHAR_INDEX['b']] == 0
    assert tmp[_CHAR_INDEX['c'] * _CHAR_SPACE_LEN + _CHAR_INDEX['c']] == 0

    assert len(tmp) == _CHAR_SPACE_LEN ** 2
    assert abs(sum(tmp) - 1.0) < 1e-6
    for value in tmp:
        assert 0.0 <= value <= 1.0

    # teste 2
    tmp = bigram_dist("http://aaa.com")
    assert tmp[_CHAR_INDEX['a'] * _CHAR_SPACE_LEN + _CHAR_INDEX['a']] == (2 / 6)

    assert len(tmp) == _CHAR_SPACE_LEN ** 2
    assert abs(sum(tmp) - 1.0) < 1e-6
    for value in tmp:
        assert 0.0 <= value <= 1.0

    # teste 3
    tmp = bigram_dist("https://abcABC.com")
    assert tmp[_CHAR_INDEX['a'] * _CHAR_SPACE_LEN + _CHAR_INDEX['b']] == (1 / 9)
    assert tmp[_CHAR_INDEX['b'] * _CHAR_SPACE_LEN + _CHAR_INDEX['c']] == (1 / 9)
    assert tmp[_CHAR_INDEX['A'] * _CHAR_SPACE_LEN + _CHAR_INDEX['B']] == (1 / 9)
    assert tmp[_CHAR_INDEX['B'] * _CHAR_SPACE_LEN + _CHAR_INDEX['C']] == (1 / 9)

    assert len(tmp) == _CHAR_SPACE_LEN ** 2
    assert abs(sum(tmp) - 1.0) < 1e-6
    for value in tmp:
        assert 0.0 <= value <= 1.0

    # teste 4
    tmp = bigram_dist("http://a.br")
    assert tmp[_CHAR_INDEX['a'] * _CHAR_SPACE_LEN + _CHAR_INDEX['.']] == (1 / 3)
    assert tmp[_CHAR_INDEX['.'] * _CHAR_SPACE_LEN + _CHAR_INDEX['b']] == (1 / 3)
    assert tmp[_CHAR_INDEX['b'] * _CHAR_SPACE_LEN + _CHAR_INDEX['r']] == (1 / 3)

    assert len(tmp) == _CHAR_SPACE_LEN ** 2
    assert abs(sum(tmp) - 1.0) < 1e-6
    for value in tmp:
        assert 0.0 <= value <= 1.0
