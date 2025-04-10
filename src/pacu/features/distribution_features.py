from urllib.parse import urlparse
from typing import *
import re
import string
import scipy
import collections
import heapq

_CHAR_SPACE = string.printable[:-6] # printable characters except whitespaces
_CHAR_SPACE_LEN = len(_CHAR_SPACE)
_CHAR_INDEX = {c: i for i, c in enumerate(_CHAR_SPACE)}

# Return character frequencies in the given URL.
def compute_frequencies(url: str) -> dict:
    return dict(collections.Counter(url))


# Helper functions
# Strip scheme and characters outside _CHAR_SPACE
def strip_url(url: str) -> str:
    url = "".join(char for char in url if char in _CHAR_SPACE)

    if (scheme := urlparse(url).scheme):
        url = re.sub(f"^{scheme}://", "", url)

    return url


# Calculates the distrubution of letters in the url 
def char_dist(url: str, freqs: list) -> list:
    url_len = len(url)
    return [freqs.get(char, 0) / url_len for char in _CHAR_SPACE]   


# distribution_unit = 1/(len(url)-1)
def bigram_dist(url: str) -> List[float]:

    url_len = len(url)
    total_bigrams = url_len - 1
    bigrams = [0.0] * (_CHAR_SPACE_LEN**2)
    distribution_unit = 1/total_bigrams
 
    for i in range(total_bigrams):
        idx = _CHAR_INDEX[url[i]] * _CHAR_SPACE_LEN + _CHAR_INDEX[url[i + 1]]
        bigrams[idx] += distribution_unit

    return bigrams


# Actual features 
# Measures how different the two distributions are using the kolmogorov–smirnov test
def kolmogorov_smirnov(url: str, calc_dist: list, dist: list) -> float:
    result = scipy.stats.ks_2samp(calc_dist, dist)
    return result[0]

# Computes the kullback–leibler divergence between two distributions
def kullback_leibler(url: str, calc_dist: list, dist: list) -> float:
    result = scipy.stats.entropy(calc_dist, dist)
    return result

# Computes the euclidean distance between two distributions
def euclidean_dist(url: str, calc_dist: list, dist: list) -> float:
    result = scipy.spatial.distance.euclidean(calc_dist, dist)
    return result

# Gets the largest individual difference between two distributions (chebyshev distance)
def cheby_shev_dist(url: str, calc_dist: list, dist: list) -> float:
    result = scipy.spatial.distance.chebyshev(calc_dist, dist)
    return result

# Computes the total absolute difference between two distributions (manhattan distance)
def manhattan_dist(url: str, calc_dist: list, dist: list) -> float:
    result = scipy.spatial.distance.cityblock(calc_dist, dist)
    return result

# Estimates the length of a Huffman encoding for the given character frequencies
def huffman(fq: dict) -> int:
    heap = list(fq.values())
    heapq.heapify(heap)

    total_length = 0

    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        merged = a + b
        total_length += merged
        heapq.heappush(heap, merged)

    return total_length
