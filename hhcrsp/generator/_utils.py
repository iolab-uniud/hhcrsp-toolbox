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

class MapDownloadError(Exception):
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
        raise MapDownloadError(f"Error downloading data: {response.status_code} {response.reason}")

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
        
