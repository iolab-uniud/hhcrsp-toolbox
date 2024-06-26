import click
from .. import __package__ as package_name
from ._utils import (
    find_appropriate_crs, 
    download_osm_data, 
    find_osm_bounds, 
    prepare_osrm_data,
    download_population_density,
    download_gadm_administrative_data
)
from .generator import generate_spatial, generate_temporal
from pathlib import Path
import shapely as sp
import platformdirs
import geopandas as gpd   
import pandas as pd
from beaupy import select_multiple
import humanize
import autopage
import pyosrm

@click.group()
def cli():
    """This is the generator group"""

@cli.command()
@click.option('--name', prompt='Your name (generator)')
def generate(name):
    click.echo(f'Hello from generator {name}')

@cli.command()
@click.argument('city', type=str, required=True)
@click.argument('radius', type=float, required=False, default=30.0)
@click.option('--name', '-n', type=str, required=False)
@click.option('--compress', '-c', is_flag=True, default=False)
@click.option('--force', '-f', is_flag=True, default=False)
def get_area(city, radius, name, compress, force):
    if name is None:
        name = f'{city}-{radius}km'
    data_dir = Path(platformdirs.user_data_dir(package_name))
    data_dir.mkdir(parents=True, exist_ok=True)
    # Geocode the city
    res = gpd.tools.geocode(city, provider='nominatim', user_agent=package_name)
    # The CRS of the result is WGS 84 (EPSG:4326), so we can set it explicitly
    res.set_crs(epsg=4326, inplace=True)
    click.echo(f'Getting area {name} centered at {res.iloc[0].address} ({res.iloc[0].geometry}), with a radius of {radius}km and saving into {data_dir}')

    # Find an appropriate CRS for the UTM zone
    crs = find_appropriate_crs(res.iloc[0].geometry.y, res.iloc[0].geometry.x)
    # Create a GeoDataFrame with the bounding box, with the appropriate CRS
    bbox = res.to_crs(crs).buffer(radius * 1000)
    bounds = sp.box(*bbox.to_crs(epsg=4326).total_bounds)
    if not force:
        click.echo(f'Checking if the data is already downloaded')
        for file in data_dir.iterdir():
            if '.osm' not in file.suffixes:
                continue
            file_bbox = find_osm_bounds(file)
            if file_bbox.contains(bounds):
                click.echo(f'OpenMap data already available in {file}, skipping download')
                return
    click.echo(f'Downloading data from OpenMap for {name}')
    download_osm_data(*bbox.to_crs(epsg=4326).total_bounds, output_filename=data_dir / f'{name}.osm', bzip=compress)

    click.echo(f'OpenMap data downloaded successfully in {data_dir}')

@cli.command()
def list_areas():
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if '.osm' in d.suffixes]
    if not files:
        click.echo('No OpenMap data available')
        return

    with autopage.AutoPager() as out:
        for d in files:
            out.write(f"{d.name} ({humanize.naturalsize(d.stat().st_size, binary=True)}) {find_osm_bounds(d).bounds}\n")

@cli.command()
def delete_areas():
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if '.osm' in d.suffixes]
    if not files:
        click.echo('No OpenMap data to delete')
        return
    selected = select_multiple([f'{d.name} ({humanize.naturalsize(d.stat().st_size, binary=True)})' for d in files], return_indices=True, pagination=True)
    if selected:
        click.confirm(f'Are you sure you want to delete {len(selected)} file(s)?', abort=True)
        freed = 0
        for i in selected:
            click.echo(f'Deleting {files[i]}')
            freed += files[i].stat().st_size
            files[i].unlink()
        click.echo(f'Freed {humanize.naturalsize(freed, binary=True)}')

@cli.command()
def create_routes():
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if '.osm' in d.suffixes]
    if not files:
        click.echo('No OpenMap data to process')
        return
    selected = select_multiple([f'{d.name} ({humanize.naturalsize(d.stat().st_size, binary=True)})' for d in files], return_indices=True, pagination=True)
    if selected:
        click.confirm(f'Are you sure you want to process {len(selected)} file(s)?', default=True, abort=True)
        for i in selected:
            click.echo(f'Processing {files[i].name}')
            prepare_osrm_data(files[i])
        click.echo(f'Processed {len(selected)} file(s)')

@cli.command()
def list_routes():
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if d.name.endswith('-routes')]
    if not files:
        click.echo('No OpenMap routes available')
        return

    with autopage.AutoPager() as out:
        for d in files:
            out.write(f"{d.name} ({humanize.naturalsize(sum(f.stat().st_size for f in d.iterdir()), binary=True)})")

@cli.command()
def delete_routes():
    data_dir = Path(platformdirs.user_data_dir(package_name))
    files = [d for d in data_dir.iterdir() if d.name.endswith('-routes')]
    if not files:
        click.echo('No OpenMap routes to delete')
        return
    selected = select_multiple([f'{d.name} ({humanize.naturalsize(sum(f.stat().st_size for f in d.iterdir()), binary=True)})' for d in files], return_indices=True, pagination=True)
    if selected:
        click.confirm(f'Are you sure you want to delete {len(selected)} directory(ies)?', abort=True)
        freed = 0
        for i in selected:
            click.echo(f'Deleting {files[i]}')
            freed += sum(f.stat().st_size for f in files[i].iterdir())
            for f in files[i].iterdir():
                f.unlink()
            files[i].rmdir()
        click.echo(f'Freed {humanize.naturalsize(freed, binary=True)}')

@cli.command()
def get_population_data():
    data_dir = Path(platformdirs.user_data_dir(package_name))
    data_dir.mkdir(parents=True, exist_ok=True)
    census_file = data_dir / 'ESTAT_Census_2021.feather'
    if census_file.exists():
        click.echo(f'Population density data already available in {data_dir}')
        return
    click.echo(f'Downloading population density data into {data_dir}')
    download_population_density(census_file)

@cli.command()
def delete_population_data():
    data_dir = Path(platformdirs.user_data_dir(package_name))
    census_file = data_dir / 'ESTAT_Census_2021.feather'
    if census_file.exists():
        click.confirm(f'Are you sure you want to delete the file {census_file.name} ({humanize.naturalsize(census_file.stat().st_size, binary=True)})?', abort=True)
        freed = census_file.stat().st_size
        census_file.unlink()
        click.echo(f'Freed {humanize.naturalsize(freed, binary=True)}')
    else:
        click.echo('No population density data to delete')

@cli.command()
@click.option('--level', '-l', type=int, default=2)
def get_administrative_data(level):
    data_dir = Path(platformdirs.user_data_dir(package_name))
    administrative_file = data_dir / f'gadm41_EU28_level_{level}.feather'
    if administrative_file.exists():
        click.echo(f'Administrative data already available in {data_dir}')
        return
    click.echo(f'Downloading administrative data (gadm4.1 level {level}) into {data_dir}')
    download_gadm_administrative_data(administrative_file, level=level)

@cli.command()
@click.option('--level', '-l', type=int, default=2)
def delete_administrative_data(level):
    data_dir = Path(platformdirs.user_data_dir(package_name))
    administrative_file = data_dir / f'gadm41_EU28_level_{level}.feather'
    if administrative_file.exists():
        click.confirm(f'Are you sure you want to delete the file {administrative_file.name} ({humanize.naturalsize(administrative_file.stat().st_size, binary=True)})?', abort=True)
        freed = administrative_file.stat().st_size
        administrative_file.unlink()
        click.echo(f'Freed {humanize.naturalsize(freed, binary=True)}')
    else:
        click.echo('No administrative data to delete')

@cli.command()
@click.option('--city', '-c', type=str, required=False)
@click.option('--radius', '-r', type=float, required=False)
@click.option('--patients', '-p', type=int)
@click.option('--departing-points', '-d', type=int)
@click.option('--')
@click.option('--no-intersect-administrative', '-n', is_flag=True, default=False)
def generate_instance(city, radius, patients, departing_points, no_intersect_administrative):
    data_dir = Path(platformdirs.user_data_dir(package_name))
    # Geocode the city
    res = gpd.tools.geocode(city, provider='nominatim', user_agent=package_name)
    # The CRS of the result is WGS 84 (EPSG:4326), so we can set it explicitly
    res.set_crs(epsg=4326, inplace=True)
    click.echo(f'Area {city} centered at {res.iloc[0].address} ({res.iloc[0].geometry}), with a radius of {radius}km')

    # Find an appropriate CRS for the UTM zone
    crs = find_appropriate_crs(res.iloc[0].geometry.y, res.iloc[0].geometry.x)
    # Create a GeoDataFrame with the bounding box, with the appropriate CRS
    bbox = res.to_crs(crs).buffer(radius * 1000)
    bounds = sp.box(*bbox.to_crs(epsg=4326).total_bounds)

    # search for the openmap data file that contains the area
    openmap_file = None
    for file in data_dir.iterdir():
        if '.osm' not in file.suffixes:
            continue
        file_bbox = find_osm_bounds(file)
        if file_bbox.contains(bounds):
            openmap_file = file
            break
    if openmap_file is None:
        click.echo(f'No OpenMap data available for {city}')
        click.prompt('Would you like to download it now?', type=bool, default=True)
        get_area(city, radius, None, False, False)
        openmap_file = data_dir / f'{city}-{radius}km.osm.bz2'
    click.echo(f'OpenMap data available in {openmap_file}')
    
    # Search for the routes data
    routes_dir = None
    ref_file_name = ".".join(openmap_file.name.split('.')[:-2])
    for file in data_dir.iterdir():        
        if file.is_dir() and (file.name.startswith(ref_file_name)) and file.name.endswith('-routes'):
            routes_dir = file
            break

    if routes_dir is None:
        click.echo('No routes data available for this area')
        click.prompt('Would you like to process it now?', type=bool, default=True)
        prepare_osrm_data(openmap_file)
        routes_dir = data_dir / f'{ref_file_name}-routes'
    routes_file = routes_dir / (ref_file_name + '.osrm')
    click.echo(f'Routes data available in {routes_dir}')
    router = pyosrm.PyOSRM(str(routes_file), algorithm='MLD')

    population_density_file = data_dir / 'ESTAT_Census_2021.feather'
    if not population_density_file.exists():
        click.echo('No population density data available')
        click.prompt('Would you like to download it now?', type=bool, default=True)
        download_population_density(population_density_file)  
    population_density = gpd.read_feather(population_density_file)
    
    # search for the closest administrative unit
    if not no_intersect_administrative:
        administrative_data_file = data_dir / 'gadm41_EU28_level_2.feather'
        if not administrative_data_file.exists():
            click.echo('No administrative data available')
            click.prompt('Would you like to download it now?', type=bool, default=True)
            get_administrative_data(2)
        administrative_data = gpd.read_feather(administrative_data_file)
        administrative_data = administrative_data.to_crs(res.crs)
        administrative_data = administrative_data[administrative_data.contains(res.iloc[0].geometry)]  
        assert administrative_data.shape[0] == 1, 'The city should be contained in a single administrative unit'       
        population_density = population_density.to_crs(administrative_data.crs)
        population_density = population_density[population_density.geometry.intersects(administrative_data.iloc[0].geometry)]
   
    # Make the datasets compatible w.r.t. crs
    population_density.to_crs(crs, inplace=True)
    res.to_crs(crs, inplace=True)
    # Select the population density within the radius
    population_density = population_density[population_density.geometry.distance(res.iloc[0].geometry) < radius * 1000]
    # Generate the instance
    click.echo('Generating instance')       
    spatial_data = generate_spatial(patients + departing_points, population_density, router)
    click.echo('Spatial distribution generated')
    instance = generate_temporal(*spatial_data)

    