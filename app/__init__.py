# app/__init__.py

# Make sure the package recognizes the modules
from . import dam_processor_app
from . import sentinel1_processing

__all__ = ['dam_processor_app', 'sentinel1_processing']