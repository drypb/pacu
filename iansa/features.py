from urllib.parse import urlparse
import pandas as pd
import re
import string
import scipy

ipv4_pattern = re.compile(r'((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}')

frequency_ptbr = [ 0.1463, 0.0104, 0.0388, 0.0499,
    0.1257, 0.0102, 0.0130, 0.0128, 0.0618, 0.0040,
    0.0002, 0.0278, 0.0474, 0.0505, 0.1073, 0.0252,
    0.0120, 0.0653, 0.0781, 0.0434, 0.0463, 0.0167,
     0.0001, 0.0021, 0.0001, 0.0047]


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
    params = [p for p in query.split('&') if p]
    return len(params)

def has_port(url: str) -> bool:
    port = urlparse(url).port
    return (port is not None)

def char_distribution(url: str) -> list:
    url=re.sub(urlparse(url).scheme, "", url) # remove scheme  
    url=re.sub("\\W|\\d", "", url) # remove non-alphabetical char

    dist = []
    for char in string.ascii_lowercase:
        dist.append(url.lower().count(char)/26)
    
    return dist;

def kolmogorov_smirnov(url: str, dist: list) -> float:
    url_dist = char_distribution(url)
    result = scipy.stats.ks_2samp(url_dist, dist)

    return result[0]

def kullback_leibler(url: str, dist: list) -> float:
    url_dist = char_distribution(url)
    result = scipy.stats.entropy(url_dist, dist)

    return result[0]

def euclidean_dist(url: str, dist: list) -> float:
    url_dist = char_distribution(url)
    result = scipy.distance.euclidean(url_dist, dist)

    return result[0]


def cheby_shev_distance(url: str, dist: list) -> float:
    url_dist = char_distribution(url)
    result = scipy.distance.chebyshev(url_dist, dist)

    return result[0]

def manhattan_dist(url: str, dist: list) -> float:
    url_dist = char_distribution(url)
    result = scipy.distance.cityblock(url_dist, dist)

    return result[0]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df['has_ip'] = df['url'].apply(has_ip)
    df['number_count'] = df['url'].apply(number_count)
    df['dash_symbol_count'] = df['url'].apply(dash_symbol_count)
    df['url_length'] = df['url'].apply(url_length)
    df['url_depth'] = df['url'].apply(url_depth)
    df['subdomain_count'] = df['url'].apply(subdomain_count)
    df['query_params_count'] = df['url'].apply(query_params_count)
    df['has_port'] = df['url'].apply(has_port)
    return df




