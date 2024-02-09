import click
from ..models import Instance, DepartingPoint
import json
import pandas as pd
from io import StringIO

@click.group()
def cli():
    """This is the validator group"""

@cli.command()
@click.argument('filename', type=click.File())
def validate(filename):
    c = Instance.model_validate_json(filename.read())
    #print(c.model_dump_json(exclude_none=True, indent=4))
    print(c.signature)

@cli.command()
@click.argument('filename', type=click.File())
@click.option('--format', type=click.Choice(['json', 'latex']), default='json', help="Output format")
@click.option('--pretty', is_flag=True, show_default=True, default=False, help="Pretty print the output")
def features(filename, format, pretty):
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
