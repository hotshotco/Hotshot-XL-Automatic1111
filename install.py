import launch
import pkg_resources

min_diffusers_version = '0.21.4'

try:
    if launch.is_installed('diffusers'):
        diffusers_version = pkg_resources.get_distribution('diffusers').version
        if diffusers_version < min_diffusers_version:
            launch.run_pip(f"install -U diffusers==0.21.4", f"Hotshot-XL requirement: changing diffusers version from {diffusers_version} to {min_diffusers_version}")
    else:
        launch.run_pip(f"install diffusers==0.21.4", 'Hotshot-XL requirement: diffusers')
except Exception as e:
    print(e)
    print(f'Warning: Failed to install diffusers, Hotshot-XL will not work.')
