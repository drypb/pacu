
import click

import extract
import train


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(extract.extract)
    cli.add_command(train.train)
    cli()
