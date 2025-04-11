from urllib.parse import urlparse
import pandas as pd
import re
import string
import scipy

ipv4_pattern = re.compile(r'((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}')

# Checks whether the URL contains an IPv4 address
def has_ip(url: str) -> bool:
    match = re.search(ipv4_pattern, url)
    return (match is not None)

# Counts how many numeric characters are in the hostname
def number_count(url: str) -> int:
    hostname = urlparse(url).hostname
    return sum(c.isdigit() for c in hostname)

# Counts how many dash ("-") symbols are in the hostname
def dash_symbol_count(url: str) -> int:
    hostname = urlparse(url).hostname
    return hostname.count('-')

# Returns the total length of the URL
def url_length(url: str) -> int:
    return len(url)

# Counts how many levels deep the path of the URL goes
def url_depth(url: str) -> int:
    path = urlparse(url).path
    segments = [s for s in path.split('/') if s] # discard empty strings
    return len(segments)

# Counts how many subdomains are present, excluding domain and TLD
def subdomain_count(url: str) -> int:
    hostname = urlparse(url).hostname
    labels = [l for l in hostname.split('.') if l]
    return (len(labels) - 2) # exclude TLD and SLD

# Returns the number of parameters in the query string
def query_params_count(url: str) -> int:
    query = urlparse(url).query
    params = [p for p in query.split('&') if p]
    return len(params)

# Checks if the URL explicitly specifies a port number
def has_port(url: str) -> bool:
    try:
        port = urlparse(url).port
    except ValueError:
        return True

    return (port is not None)
