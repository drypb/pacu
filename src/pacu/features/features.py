from pacu.features.lexical_features import *
from pacu.features.distribution_features import *
from pacu.features.distributions import *

import pandas as pd

def process_row(row: pd.Series) -> pd.Series:
    url = row["url"]
    url_s = strip_url(url)
    fq = compute_frequencies(url_s)
    chardist = char_dist(url_s, fq)
    return pd.Series({
        "url"               : url,
        "has_ip"            : has_ip(url),
        "number_count"      : number_count(url),
        "dash_symbol_count" : dash_symbol_count(url),
        "url_length"        : url_length(url),
        "url_depth"         : url_depth(url),
        "subdomain_count"   : subdomain_count(url),
        "query_params_count": query_params_count(url),
        "has_port"          : has_port(url),
        "ks_char"           : kolmogorov_smirnov(url_s, chardist, frequency_char_ptbr),
        "kl_char"           : kullback_leibler(url_s, chardist, frequency_char_ptbr),
        "eucli_char"        : euclidean_dist(url_s, chardist, frequency_char_ptbr),
        "cs_char"           : cheby_shev_dist(url_s, chardist, frequency_char_ptbr),
        "man_char"          : manhattan_dist(url_s, chardist, frequency_char_ptbr),
        "huffman"           : huffman(fq),
        "label"             : row["label"]
    })


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(process_row, axis=1)
