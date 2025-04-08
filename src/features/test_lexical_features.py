from lexical_features import *

def test_has_ip():
    assert has_ip("http://192.168.1.1") == True
    assert has_ip("https://example.com") == False
    assert has_ip("ftp://10.0.0.1") == True
    assert has_ip("http://example.com") == False
    assert has_ip("http://200.17.210.55/path/to/something/") == True
    assert has_ip("http://lalala.lelele.shop.br/192.168.1.1/") == True
    assert has_ip("http://192.168.1.1.website.com.br/192.168.1.1/") == True
    assert has_ip("http://example.com/?r=1.1.1.1") == True

def test_number_count():
    assert number_count("http://example123.com") == 3
    assert number_count("https://google.com") == 0
    assert number_count("http://192.168.1.1") == 8
    assert number_count("ftp://test-123.com") == 3

def test_dash_symbol_count():
    assert dash_symbol_count("https://example.com") == 0
    assert dash_symbol_count("http://sub-domain.example.com") == 1
    assert dash_symbol_count("ftp://test-si-te.example.com") == 2

def test_url_length():
    assert url_length("http://example.com") == 18
    assert url_length("https://www.example.com/test") == 28
    assert url_length("ftp://file.example.com") == 22
    assert url_length("http://localhost:8080") == 21

def test_url_depth():
    assert url_depth("http://example.com/one/two/three") == 3
    assert url_depth("https://example.com") == 0
    assert url_depth("http://example.com/a/b/c/d/e") == 5
    assert url_depth("ftp://example.com/") == 0
    assert url_depth("http://stress.testing.com//////////////////") == 0

def test_subdomain_count():
    assert subdomain_count("http://sub.domain.example.com") == 2
    assert subdomain_count("https://example.com") == 0
    assert subdomain_count("ftp://test.subdomain.example.com") == 2
    assert subdomain_count("http://www.example.com") == 1

def test_query_params_count():
    assert query_params_count("http://example.com?param1=value1&param2=value2") == 2
    assert query_params_count("https://example.com") == 0
    assert query_params_count("http://example.com?param=value") == 1
    assert query_params_count("http://example.com?") == 0

def test_has_port():
    assert has_port("http://example.com:8080") == True
    assert has_port("https://example.com") == False
    assert has_port("ftp://example.com:21") == True
    assert has_port("http://localhost") == False
    assert has_port("http://my.website.com:7272/path/to/somt") == True

def test_lexical_features():
    test_has_ip()
    test_number_count()
    test_dash_symbol_count()
    test_url_length()
    test_url_depth()
    test_subdomain_count()
    test_query_params_count()
    test_has_port()
