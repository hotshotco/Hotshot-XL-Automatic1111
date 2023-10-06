# Copyright 2023 Natural Synthetics Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

from diffusers.utils import (
    BaseOutput,
)

@dataclass
class HotshotPipelineXLOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]