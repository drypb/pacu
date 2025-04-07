
import click

import extract
import train


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_argument(extract.extract)
    cli.add_argument(train.train)
    cli()
