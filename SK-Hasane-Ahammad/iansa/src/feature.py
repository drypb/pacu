
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
            pattern = r'((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4}'
            match = re.search(pattern, url)

            return (match is not None)

        # bernas
        def numberSig():
            pass

        # bernas
        def symbolSig():
            pass

        def urlLen():
            pass

        def urlDepth():
            pass

        def subdomainSig():
            pass 

        def argSig():
            pass

        # bernas
        def randomSig():
            pass

