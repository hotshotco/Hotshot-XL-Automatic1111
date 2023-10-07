print("Hotshot-XL install.py")

import logging
import subprocess

# Set up logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Hotshot-XL Ext: %(message)s')
logger = logging.getLogger()

min_diffusers_version = "0.21.4"

# Depending on the Python version, we might need the external package or the standard library one
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

def is_diffusers_installed():
    try:
        # Check if 'diffusers' is installed
        installed_version = version("diffusers")
        return installed_version
    except PackageNotFoundError:
        return None

installed_version = is_diffusers_installed()

if installed_version is None:
    logger.info("Diffusers package is not installed. Installing now...")
    subprocess.run(f"pip install diffusers=={min_diffusers_version} --upgrade".split(" "))
elif tuple(map(int, installed_version.split('.'))) < tuple(map(int, min_diffusers_version.split('.'))):
    logger.info(f"Diffusers package version {installed_version} is less than required {min_diffusers_version}. Upgrading now...")
    subprocess.run(f"pip install diffusers=={min_diffusers_version} --upgrade".split(" "))
else:
    logger.info(f"Diffusers package version {installed_version} is up to date.")
