from urllib.parse import urlparse
import pandas as pd
import re
import string
import scipy

import lexical_features as lf
import distribution_features as dstf
import distributions as dsts


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df['has_ip'] = df['url'].apply(has_ip)
    df['number_count'] = df['url'].apply(number_count)
    df['dash_symbol_count'] = df['url'].apply(dash_symbol_count)
    df['url_length'] = df['url'].apply(url_length)
    df['url_depth'] = df['url'].apply(url_depth)
    df['subdomain_count'] = df['url'].apply(subdomain_count)
    df['query_params_count'] = df['url'].apply(query_params_count)
    df['has_port'] = df['url'].apply(has_port)

    df['ks_char'] = df['url'].apply(lambda x: dstf.kolmogorov_smirnov(x, dstf.char_dist, dsts.frequency_char_ptbr))
    df['kl_char'] = df['url'].apply(lambda x: dstf.kullback_leibler(x, dstf.char_dist, dsts.frequency_char_ptbr))
    df['eucli_char'] = df['url'].apply(lambda x: dstf.euclidean_dist(x, dstf.char_dist, dsts.frequency_char_ptbr))
    df['eucli_char'] = df['url'].apply(lambda x: dstf.euclidean_dist(x, dstf.char_dist, dsts.frequency_char_ptbr))
    df['cs_char'] = df['url'].apply(lambda x: dstf.cheby_shev_dist(x, dstf.char_dist, dsts.frequency_char_ptbr))
    df['man_char'] = df['url'].apply(lambda x: dstf.manhattan_dist(x, dstf.char_dist, dsts.frequency_char_ptbr))
    
    df['ks_bigram'] = df['url'].apply(lambda x: dstf.kolmogorov_smirnov(x, dstf.bigram_dist, dsts.frequency_bigram_unlabeled))
    df['kl_bigram'] = df['url'].apply(lambda x: dstf.kullback_leibler(x, dstf.bigram_dist, dsts.frequency_bigram_unlabeled))
    df['eucli_bigram'] = df['url'].apply(lambda x: dstf.euclidean_dist(x, dstf.bigram_dist, dsts.frequency_bigram_unlabeled))
    df['eucli_bigram'] = df['url'].apply(lambda x: dstf.euclidean_dist(x, dstf.bigram_dist, dsts.frequency_bigram_unlabeled))
    df['cs_bigram'] = df['url'].apply(lambda x: dstf.cheby_shev_dist(x, dstf.bigram_dist, dsts.frequency_bigram_unlabeled))
    df['man_bigram'] = df['url'].apply(lambda x: dstf.manhattan_dist(x, dstf.bigram_dist, dsts.frequency_bigram_unlabeled))
    
   
    return df




