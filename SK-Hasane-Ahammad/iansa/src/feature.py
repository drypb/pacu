
import pathlib
import pandas
import csv


class Features:

    path: str
    csvf: pandas.DataFrame

    def __init__(self, path: str):
        self.csvf = pd.read_csv(path)
        self.path = path

    
    def extract(self):
        
        def hasIP():
            pass

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

