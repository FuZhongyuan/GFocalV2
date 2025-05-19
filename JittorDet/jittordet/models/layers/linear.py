import jittor.nn as nn

from GFocalV2Jittor.engine import MODELS

MODELS.register_module('Linear', module=nn.Linear)
