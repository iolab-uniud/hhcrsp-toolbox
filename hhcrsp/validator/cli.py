import click

@click.group()
def cli():
    """This is the validator group"""

@cli.command()
@click.option('--name', prompt='Your name (validator)')
def validate(name):
    click.echo(f'Hello from validator {name}')