import requests
from pyproj import CRS
from math import floor
import xml.etree.ElementTree as ET
import shapely as sp
import subprocess
from pathlib import Path
import bz2
import click
import tempfile
import zipfile
import geopandas as gpd
import pandas as pd
import country_converter as coco 
import io

class DataDownloadError(Exception):
    pass

class OSRMProcessingError(Exception):
    pass

def find_appropriate_crs(latitude, longitude):
    """
    Find an appropriate CRS based on latitude and longitude.
    
    Parameters:
    - latitude (float): Latitude in decimal degrees.
    - longitude (float): Longitude in decimal degrees.
    
    Returns:
    - CRS: A pyproj CRS object for the appropriate UTM zone.
    """

    # Calculate the UTM zone number for a given longitude.
    utm_zone = floor((longitude + 180) / 6) + 1
    hemisphere = 'north' if latitude >= 0 else 'south'
    crs = CRS(f"EPSG:326{utm_zone}") if hemisphere == 'north' else CRS(f"EPSG:327{utm_zone}")
    return crs

def download_osm_data(west : float, south : float, east : float, north : float, output_filename : Path=Path('map.osm'), bzip : bool=False):
    """
    Download OSM data for a specified bounding box and save it as a .pbf file.

    Parameters:
    - west (float): Western longitude of the bounding box.
    - south (float): Southern latitude of the bounding box.
    - east (float): Eastern longitude of the bounding box.
    - north (float): Northern latitude of the bounding box.
    - output_filename (Path): Name of the file to save the downloaded data to.
    - bzip (bool): Whether to compress the output file with bzip2.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:xml][maxsize:2000000000][timeout:60][bbox:{south},{west},{north},{east}];
    (node;way;rel;);
    out meta;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    if response.status_code == 200:
        if bzip:
            if output_filename.suffix != '.bz2':
                output_filename = output_filename.with_suffix(output_filename.suffix + '.bz2')
            open_func = bz2.open
        else:
            open_func = open
        with open_func(output_filename, 'wb') as file:
            file.write(response.content)
    else:
        raise DataDownloadError(f"Error downloading data from {overpass_url}: {response.status_code} {response.reason}")

def find_osm_bounds(filename : Path):
    if '.bz2' in filename.suffixes:
        open_func = bz2.open
    else:
        open_func = open    
    with open_func(filename, 'rb') as f:
        # Create an iterator for the XML file
        context = ET.iterparse(f, events=('start', 'end'))

        # Skip the root element
        _, root = next(context)

        # Process each element in the XML file
        for event, element in context:
            if event == 'end' and element.tag == 'bounds':
                # Process the <bounds> element
                res = element.attrib
                return sp.box(float(res['minlon']), 
                              float(res['minlat']), 
                              float(res['maxlon']), 
                              float(res['maxlat']))

            # Clear the processed elements from memory
            root.clear()

def run_osrm_process(command : str, args : list[str]):
    """
    Runs an OSRM backend process with the given command and arguments.

    Parameters:
    - command (str): The OSRM backend command to run (e.g., 'osrm-extract').
    - args (list): A list of arguments to pass to the command.
    """
    try:
        subprocess.run([command] + args, check=True)
    except subprocess.CalledProcessError as e:
        raise OSRMProcessingError(f"Error running {command}: {e}")

def prepare_osrm_data(osm_file_path : Path, profile_path : Path=None):
    """
    Prepares an OSM .pbf file for routing with OSRM by extracting, partitioning, and customizing the data.

    Parameters:
    - osm_file_path (pathlib.Path): The path to the OSM file.
    - profile_path (pathlib.Path): The path to the OSRM profile file (e.g., car.lua).
    """
    # due to the way osrm-extract works, we need to link the pbf_file_path to a file in a temporary directory
    # so that the file can be found by the osrm-extract process
        
    with tempfile.TemporaryDirectory() as temp_dir:
        osm_file_temp_path = Path(temp_dir) / osm_file_path.name
        osm_file_temp_path.symlink_to(osm_file_path)
        click.echo(f'Processing {osm_file_temp_path}')

        if profile_path is None:
            profile_path = Path(__file__).parent.parent / 'osrm_profiles' / 'car.lua'
        # Extract
        run_osrm_process('osrm-extract', ['-p', profile_path, osm_file_temp_path])
        
        # The output file from extraction will have the same name but with .osrm extension
        base_osm_filename = osm_file_path.name.split('.osm')[0]
        osrm_file_path = Path(temp_dir) / (base_osm_filename + '.osrm')
        
        # Partition
        run_osrm_process('osrm-partition', [osrm_file_path])
        
        # Customize
        run_osrm_process('osrm-customize', [osrm_file_path])

        # Move the file to the original directory
        target_dir = osm_file_path.parent / (base_osm_filename + '-routes')
        target_dir.mkdir(exist_ok=True, parents=True)
        for file in Path(temp_dir).iterdir():
            if '.osrm' in file.suffixes:
                file.rename(target_dir / file.name)

        click.echo(f"Data preparation complete. Ready for routing with data in {target_dir}")
        
def download_population_density(filename : Path):
    url = 'https://gisco-services.ec.europa.eu/census/2021/Eurostat_Census-GRID_2021_V1-0.zip'
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.TemporaryDirectory() as tmp_dir, open(Path(tmp_dir) / 'census_data.zip', 'wb') as file:
            file.write(response.content)
            file.seek(0)
            click.echo(f'Downloaded Eurostat census data')
            with zipfile.ZipFile(file.name, 'r') as zip_ref:
                for file in zip_ref.namelist():
                    if '.gpkg' in file:
                    #    (Path(tmp_dir) / file).rename(filename)
                        click.echo(f'Reading and transforming {file}')
                        gdf = gpd.read_file(zip_ref.open(file))
                        # Filter out rows with no population
                        gdf = gdf[gdf.OBS_VALUE_T > 0.0]
                        click.echo(f'Writing to {filename}')
                        gdf.to_feather(Path(tmp_dir) / filename, compression='zstd')
                        break
    else:
        raise DataDownloadError(f"Error downloading census data from {url}: {response.status_code} {response.reason}")

def download_gisco_administrative_data(year='2021', resolution='01M', level='2', epsg_proj='3035'):
    """
    Fetch local administrative units from the GISCO API.
    
    Parameters:
    - year: The year of the data. Default is '2021'.
    - resolution: The resolution of the geographical data. Options include '01M', '03M', '10M', '20M', and '60M'. Default is '01M'.
    - level: The administrative level. Level '2' for NUTS 2 regions, and '3' for LAU (Local Administrative Units). Default is '2'.
    
    Returns:
    - A JSON object containing the requested geographical data.
    """
    # GISCO API endpoint for administrative units
    base_url = 'https://gisco-services.ec.europa.eu/distribution/v2'
    if level == '2':
        data_type = 'nuts'
        file_name = 'NUTS_RG'
    elif level == '3':
        data_type = 'lau'
        file_name = 'LAU_RG'
    else:
        raise ValueError("Unsupported level. Choose '2' for NUTS 2 or '3' for LAU.")
    
    # Constructing the URL based on the parameters
    request_url = f'{base_url}/{data_type}/geojson/{file_name}_{resolution}_{year}_{epsg_proj}.geojson'
    
    click.echo("Getting administrative data from GISCO")
    gdf = gpd.read_file(request_url)

    gdf.to_feather('prova.feather', compression='zstd')

def download_gadm_administrative_data(filename : Path, level : int=2):
    """
    Fetch local administrative units from the GADM API.
    
    Parameters:
    - level: The administrative level. Default is 2.
    """
    gdf = gpd.GeoDataFrame()
    cc = coco.CountryConverter()
    for country in list(cc.EU28as('ISO3')['ISO3']):
        gadm_url = f'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{country}_{level}.json.zip'
        res = requests.get(gadm_url)
        if res.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                for file in z.namelist():
                    if '.json' in file:
                        gdf = pd.concat([gdf, gpd.read_file(z.open(file))])
        else:
            click.echo(f'Skipping country {country} due to error {res.status_code} {res.reason}')

    gdf.to_feather(filename, compression='zstd')