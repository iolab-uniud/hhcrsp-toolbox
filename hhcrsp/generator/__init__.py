try:
    # TODO: check also for availability of osrm-backend library
    import pyosrm
    import geopandas
    _HAS_GENERATOR = True
except ImportError as e:
    _HAS_GENERATOR = False
