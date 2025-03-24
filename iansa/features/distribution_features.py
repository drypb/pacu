from urllib.parse import urlparse
from typing import Callable
import re
import string
import scipy


# Helper functions


# Replace all special symbols and numbers with a chosen replacement
def strip_url(url: str, replacement_char: str = "") -> str:
    url=re.sub(urlparse(url).scheme, "", url) # remove scheme  
    url=re.sub("\\W|\\d", replacement_char, url) # remove non-alphabetical char

    return url

# Calculates the distrubution of letters in the url 
def char_dist(url: str) -> list:
    url=strip_url(url)

    dist = []
    for char in string.ascii_lowercase:
        dist.append(url.lower().count()/url.length()) 
    
    return dist;


# Calculates the number of bigrams in the url
def bigram_count(url: str, delimeter: str = " ") -> int:
    bag_of_words = url.split(delimeter)

    return sum(list(map(lambda x: x.length() - 1, bag_of_words)))

# Calculates the index of a given bigram in a vector 26*26 elements long
def bigram_index(bigram: str) -> int:
    return ord(bigram[0].lower())%0x61*26 + ord(bigram[1].lower())

# Calculate the bigram distribution for a given url
def bigram_dist(url: str) -> list:
    delim = "."

    url = strip_url(url, delim)    
    bigrams = [0] * 26 * 26

    distribution_unit = 1/bigram_count(url, delim)
 
    for i in enumerate(url[:-1]):
        if delim not in url[i:i+1]: 
            bigrams[bigram_index(url[i:i+1])] += distribution_unit

    return bigrams

# Actual features 

def kolmogorov_smirnov(url: str, calc_dist: Callable[[str],list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.stats.ks_2samp(url_dist, dist)

    return result[0]

def kullback_leibler(url: str, calc_dist: Callable[str,list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.stats.entropy(url_dist, dist)

    return result[0]

def euclidean_dist(url: str, calc_dist: Callable[str,list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.spatial.distance.euclidean(url_dist, dist)

    return result

def cheby_shev_distance(url: str, calc_dist: Callable[str,list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.spatial.distance.chebyshev(url_dist, dist)

    return result

def manhattan_dist(url: str, calc_dist: Callable[str,list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.spatial.distance.cityblock(url_dist, dist)

    return result





