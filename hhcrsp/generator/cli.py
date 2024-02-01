import click

@click.group()
def cli():
    """This is the generator group"""

@cli.command()
@click.pass_context
@click.option('--name', prompt='Your name (generator)')
def generate(ctx, name):
    click.echo(f'Hello from generator {name}')