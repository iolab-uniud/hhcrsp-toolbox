import click
from .. generator import _HAS_GENERATOR

commands = []

if _HAS_GENERATOR:
    from .. generator.cli import cli as cli_generator
    commands.append(cli_generator)
else:
    @click.group()
    def cli_generator():
        pass
    @cli_generator.command()
    def generate():
        click.secho('⚠️ Generator not available: you should install the hhcrsp-toolbox with the generator extra (requires osrm library to be installed)', fg='magenta')
    commands.append(cli_generator)

from .. validator.cli import cli as cli_validator
commands.append(cli_validator)
 
def main_cli():
    cli = click.CommandCollection(sources=commands)
    cli(obj={})
