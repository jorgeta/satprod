import yaml
from munch import Munch
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

def read_yaml(path: str) -> Munch:
    with open(path) as f:
        return Munch.fromDict(yaml.load(f, yaml.SafeLoader))

class ImgType(Enum):
    SAT = 'sat'
    FB_DENSE = 'fb_dense'
    LK_SPARSE = 'lk_sparse'
    LK_SPARSEMASK = 'lk_sparsemask'
    LK_DENSE = 'lk_dense'
    DTVL1_DENSE = 'dtvl1_dense'
    RLOF_DENSE = 'rlof_dense'

@dataclass
class TimeInterval:
    start: datetime
    stop: datetime