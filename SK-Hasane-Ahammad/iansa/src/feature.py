
from urllib.parse import urlparse
import pandas
import re

class Features:

    path: str
    csvf: pandas.DataFrame

    def __init__(self, path: str):
        self.csvf = pandas.read_csv(path)
        self.path = path

    
    def extract(self):
        
        def hasIP(url: str) -> bool:
            pattern = r'((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}' # ipv4 pattern
            match = re.search(pattern, url)

            return (match is not None)

        # bernas
        def numberSig():
            pass

        # bernas
        def symbolSig():
            pass

        def urlLen(url: str) -> int:
            return len(url)

        def urlDepth(url: str) -> int:
            path = urlparse(url).path
            segments = [s for s in path.split('/') if s] # discard empty strings

            return len(segments)

        def subdomainCount(url: str) -> int:
            hostname = urlparse(url).hostname
            labels = [l for l in hostname.split('.') if l]

            return (len(labels) - 2) # exclude TLD and SLD

        def argSig():
            pass

        # bernas
        def randomSig():
            pass

