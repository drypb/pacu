from urllib.parse import urlparse
import pandas as pd
import re
import string
import scipy

ipv4_pattern = re.compile(r'((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}')

def has_ip(url: str) -> bool:
    match = re.search(ipv4_pattern, url)
    return (match is not None)

def number_count(url: str) -> int:
    hostname = urlparse(url).hostname
    return sum(c.isdigit() for c in hostname)

def dash_symbol_count(url: str) -> int:
    hostname = urlparse(url).hostname
    return hostname.count('-')

def url_length(url: str) -> int:
    return len(url)

def url_depth(url: str) -> int:
    path = urlparse(url).path
    segments = [s for s in path.split('/') if s] # discard empty strings
    return len(segments)

def subdomain_count(url: str) -> int:
    hostname = urlparse(url).hostname
    labels = [l for l in hostname.split('.') if l]
    return (len(labels) - 2) # exclude TLD and SLD

def query_params_count(url: str) -> int:
    query = urlparse(url).query
