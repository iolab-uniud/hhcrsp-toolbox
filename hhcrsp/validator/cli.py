import click
from ..models import Instance, Solution
import json
import pandas as pd
from pydantic import ValidationError

@click.group()
def cli():
    """This is the validator group"""

@cli.command()
@click.argument('filename', type=click.File())
def validate_instance(filename):
    try:
        c = Instance.model_validate_json(filename.read())    
        click.secho(f"Validation OK, instance signature: {c.signature}", fg='green')
    except ValidationError as e:
        click.secho(f"{e}", fg='red')

@cli.command()
@click.argument('filename', type=click.File())
@click.option('--format', type=click.Choice(['json', 'latex']), default='json', help="Output format")
@click.option('--pretty', is_flag=True, show_default=True, default=False, help="Pretty print the output")
def instance_features(filename, format, pretty):
    c = Instance.model_validate_json(filename.read())
    if format == 'json':
        if not pretty:
            print(json.dumps(c.features))
        else:
            class RoundingFloat(float):
                __repr__ = staticmethod(lambda x: f'{x:.3f}')
            json.encoder.c_make_encoder = None
            json.encoder.float = RoundingFloat
            print(json.dumps(c.features, indent=4))
    elif format == 'latex':        
        reformed_dict = {} 
        for outerKey, innerDict in c.features.items(): 
            if type(innerDict) == dict:
                for innerKey, values in innerDict.items(): 
                    reformed_dict[(outerKey, innerKey)] = [values]
            else:
                reformed_dict[(outerKey, )] = [innerDict]
        df = pd.DataFrame(reformed_dict)
        print(df.to_latex(index=False))

@cli.command()
@click.argument('instance-filename', type=click.File())
@click.argument('solution-filename', type=click.File())
def validate_solution(instance_filename, solution_filename):
    try:
        i = Instance.model_validate_json(instance_filename.read())
        s = Solution.model_validate_json(solution_filename.read())    
        click.secho(f"Validation OK, solution", fg='green')
        try:
            s.check_validity(i)
        except Exception as e:
            raise ValidationError(e)
        print(s.compute_costs(i))
    except ValidationError as e:
        click.secho(f"{e}", fg='red')

@cli.command()
@click.argument('instance-filename', type=click.File())
@click.argument('solution-filename', type=click.File())
@click.option('--output', '-o', type=click.File('w'), default='-', help="Output file")
def convert_solution(instance_filename, solution_filename, output):
    try:
        i = Instance.model_validate_json(instance_filename.read())
        s = Solution.model_validate_json(solution_filename.read())    
        click.secho(f"Validation OK, solution", fg='green')
        try:
            s.check_validity(i)
        except Exception as e:
            raise ValidationError(e)
        output.write(s.model_dump_json(exclude_unset=True, indent=4))
    except ValidationError as e:
        click.secho(f"{e}", fg='red')

    
@cli.command()
@click.argument('instance-filename', type=click.File())
@click.argument('solution-filename', type=click.File())
@click.option('--output', '-o', type=click.Path(), help="Output file")
def plot_solution(instance_filename, solution_filename, output):
    from .plot import plot
    i = Instance.model_validate_json(instance_filename.read())
    s = Solution.model_validate_json(solution_filename.read())    
    s.check_validity(i)
    plt = plot(i, s)
    if not output:
        plt.show()
    else:
        plt.savefig(output)