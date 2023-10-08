print("Hotshot-XL install.py")

import logging
import os
from modules import scripts
import packages

os.makedirs(os.path.join(scripts.basedir(), "model"), exist_ok=True)

logger = logging.getLogger()

# if not packages.is_diffusers_installed():
#     packages.install_diffusers()