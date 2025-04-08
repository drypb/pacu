
import features.extractor as fe
import click


@click.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out" , default="out.csv")
def extract(path: str, out: str) -> None:
    f = fe.FeatureExtractor(path) 
    f.extract()
    f.export(out)
