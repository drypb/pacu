
import click

from . import extract
from . import train


@click.group()
def cli():
    pass


cli.add_command(extract.extract)
cli.add_command(train.train)
cli()
