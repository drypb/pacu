
from urllib.parse import urlparse
from typing import *
import re
import string
import scipy

_CHAR_SPACE = string.printable[:-6] # printable characters except whitespaces
_CHAR_SPACE_LEN = len(_CHAR_SPACE)
_CHAR_INDEX = {c: i for i, c in enumerate(_CHAR_SPACE)}


# Helper functions
# Strip scheme and characters outside _CHAR_SPACE
def strip_url(url: str) -> str:
    url = "".join(char for char in url if char in _CHAR_SPACE)

    if (scheme := urlparse(url).scheme):
        url = re.sub(f"^{scheme}://", "", url)

    return url


# Calculates the distrubution of letters in the url 
def char_dist(url: str) -> list:
    url = strip_url(url)
    url_len = len(url)
    dist = []
    for char in _CHAR_SPACE:
        dist.append(url.count(char)/url_len) 
    
    return dist


# distribution_unit = 1/(len(url)-1)
def bigram_dist(url: str) -> List[float]:

    url = strip_url(url)    
    url_len = len(url)
    total_bigrams = url_len - 1
    bigrams = [0.0] * (_CHAR_SPACE_LEN**2)
    distribution_unit = 1/total_bigrams
 
    for i in range(total_bigrams):
        idx = _CHAR_INDEX[url[i]] * _CHAR_SPACE_LEN + _CHAR_INDEX[url[i + 1]]
        bigrams[idx] += distribution_unit

    return bigrams


# Actual features 
def kolmogorov_smirnov(url: str, calc_dist: Callable[[str],list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.stats.ks_2samp(url_dist, dist)

    return result[0]

def kullback_leibler(url: str, calc_dist: Callable[str,list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.stats.entropy(url_dist, dist)

    return result

def euclidean_dist(url: str, calc_dist: Callable[str,list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.spatial.distance.euclidean(url_dist, dist)

    return result

def cheby_shev_dist(url: str, calc_dist: Callable[str,list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.spatial.distance.chebyshev(url_dist, dist)

    return result

def manhattan_dist(url: str, calc_dist: Callable[str,list], dist: list) -> float:
    url_dist = calc_dist(url)
    result = scipy.spatial.distance.cityblock(url_dist, dist)

    return result





