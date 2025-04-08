from urllib.parse import urlparse
import pandas as pd
import re
import string
import scipy

from features.lexical_features import *
from features.distribution_features import *
from features.distributions import *

def extract_features(df: pd.DataFrame) -> pd.DataFrame:

    df["has_ip"] = df["url"].apply(has_ip)
    print("has_ip completed.")

    df["number_count"] = df["url"].apply(number_count)
    print("number_count completed.")

    df["dash_symbol_count"] = df["url"].apply(dash_symbol_count)
    print("dash_symbol_count completed.")

    df["url_length"] = df["url"].apply(url_length)
    print("url_length complted.")

    df["url_depth"] = df["url"].apply(url_depth)
    print("url_depth completed.")

    df["subdomain_count"] = df["url"].apply(subdomain_count)
    print("subdomain_count completed.")

    df["query_params_count"] = df["url"].apply(query_params_count)
    print("subdomain_count completed.")

    df["has_port"] = df["url"].apply(has_port)
    print("has_port completed.")

    df["ks_char"] = df["url"].apply(lambda x: kolmogorov_smirnov(x, char_dist, frequency_char_ptbr))
    print("ks_char completed.")

    df["kl_char"] = df["url"].apply(lambda x: kullback_leibler(x, char_dist, frequency_char_ptbr))
    print("kl_char completed.")

    df["eucli_char"] = df["url"].apply(lambda x: euclidean_dist(x, char_dist, frequency_char_ptbr))
    print("eucli_char completed.")

    df["cs_char"] = df["url"].apply(lambda x: cheby_shev_dist(x, char_dist, frequency_char_ptbr))
    print("cs_char completed.")

    df["man_char"] = df["url"].apply(lambda x: manhattan_dist(x, char_dist, frequency_char_ptbr))
    print("man_char completed.")
    
    return df