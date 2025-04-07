from distribution_features import *

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
    assert strip_url("http://example.com/r?=http://redirect.org") == "example.com/r?=http://redirect.org"

def test_char_dist():
    # URL simples com poucas letras
    url1 = "https://www.example.com"
    result1 = char_dist(url1)
    print()
    assert len(result1) == 94
    
    # URL com caracteres repetidos
    url2 = "http://aaaaa.com"
    result2 = char_dist(url2)
    assert result2[0] > 0.0

    # URL com letras mistas
    url3 = "ftp://abcABC.com"
    result3 = char_dist(url3)
    # As letras a, b, c, A, B, C terão algumas contagens
    assert result3[0] > 0  # A letra 'a' deve aparecer
    assert result3[26] > 0  # A letra 'A' deve aparecer
    assert result3[1] > 0  # A letra 'b' deve aparecer
    assert result3[27] > 0  # A letra 'B' deve aparecer
    assert result3[2] > 0  # A letra 'c' deve aparecer
    assert result3[28] > 0  # A letra 'C' deve aparecer
    
    # URL com letras e outros caracteres não alfabéticos
    url5 = "http://abc-123.com"
    result5 = char_dist(url5)
    # A letra 'a', 'b' e 'c' terão uma certa contagem
    assert result5[0] > 0  # A letra 'a' deve aparecer
    assert result5[1] > 0  # A letra 'b' deve aparecer
    assert result5[2] > 0  # A letra 'c' deve aparecer
    assert result5[53] > 0  # A letra '1' deve aparecer
    assert result5[54] > 0  # A letra '2' deve aparecer
    assert result5[55] > 0  # A letra '3' deve aparecer
    
    # URL vazia
    url6 = "http://"
    result6 = char_dist(url6)
    # Para uma URL vazia, todas as distribuições devem ser zero
    assert all((not r == 0) for r in result6)

def test_bigram_dist():
    # URL simples com poucas letras
    url1 = "https://abc.com"
    result1 = bigram_dist(url1)
    assert len(result1) == _CHAR_SPACE_LEN ** 2  # Deve ter 94 * 94 distribuições de bigramas

    # URL com caracteres repetidos
    url2 = "http://aaa.com"
    result2 = bigram_dist(url2)
    # O par 'a' seguido de 'a' deve aparecer mais vezes, as outras combinações devem ter 0.0
    assert result2[_CHAR_INDEX['a'] * _CHAR_SPACE_LEN + _CHAR_INDEX['a']] > 0

    # URL com letras mistas
    url3 = "https://abcABC.com"
    result3 = bigram_dist(url3)
    assert result3[_CHAR_INDEX['a'] * _CHAR_SPACE_LEN + _CHAR_INDEX['b']] > 0
    assert result3[_CHAR_INDEX['b'] * _CHAR_SPACE_LEN + _CHAR_INDEX['c']] > 0
    assert result3[_CHAR_INDEX['A'] * _CHAR_SPACE_LEN + _CHAR_INDEX['B']] > 0
    assert result3[_CHAR_INDEX['B'] * _CHAR_SPACE_LEN + _CHAR_INDEX['C']] > 0

    # URL vazia
    url4 = "http://"
    result4 = bigram_dist(url4)
    assert not all(val != 0 for val in result4)

    # URL com apenas um caractere
    url5 = "http://x"
    result5 = bigram_dist(url5)
    assert not all(val != 0 for val in result5)

def test_distribution_features():
    test_strip_url()
    test_char_dist()
    test_bigram_dist()
