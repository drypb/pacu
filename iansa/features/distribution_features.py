from urllib.parse import urlparse
from typing import Callable
import re
import string
import scipy


# Helper functions


# Replace all special symbols and numbers with a chosen replacement
def strip_url(url: str, replacement_char: str = "") -> str:
    url=re.sub(urlparse(url).scheme, "", url) # remove scheme  
    url=re.sub("://", "", url)
    return url

# Calculates the distrubution of letters in the url 
def char_dist(url: str) -> list:
    url=strip_url(url)

    dist = []
    for char in string.printable[:-6]:
        dist.append(url.count(char)/len(url)) 
    
    return dist;

# Calculate the bigram distribution for a given url
def bigram_dist(url: str) -> list:
    url = strip_url(url)    
    bigrams = [0]*len(string.printable)**2 

    distribution_unit = 1/(len(url)-1)
 
    for i, char in enumerate(url[:-1]):
        idx1 = string.printable.find(url[i])*len(string.printable[:-6])
        idx2 = string.printable.find(url[i+1])
        bigrams[idx1 + idx2] += distribution_unit

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





