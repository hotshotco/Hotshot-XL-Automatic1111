import subprocess

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

def install_diffusers():
    installed_version = is_diffusers_installed()

    if installed_version is None:
        print("Diffusers package is not installed. Installing now...")
        subprocess.run(f"pip install diffusers=={min_diffusers_version} --upgrade".split(" "))
    elif tuple(map(int, installed_version.split('.'))) < tuple(map(int, min_diffusers_version.split('.'))):
        print(f"Diffusers package version {installed_version} is less than required {min_diffusers_version}. Upgrading now...")
        subprocess.run(f"pip install diffusers=={min_diffusers_version} --upgrade".split(" "))
    else:
        print(f"Diffusers package version {installed_version} is up to date.")

