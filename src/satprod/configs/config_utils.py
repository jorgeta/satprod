import yaml
from munch import Munch
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

def read_yaml(path: str) -> Munch:
    """Read yaml file to a Munch object.
    Munch object is a dictionary that provides attribute-style access.
    Used for yaml configs.
    Args:
        path: Path to the yaml file.
    Returns:
        Munch object.
    """
    with open(path) as f:
        return Munch.fromDict(yaml.load(f, yaml.SafeLoader))

class ImgType(Enum):
    SAT = 'sat'
    FB_DENSE = 'fb_dense'
    LK_SPARSE = 'lk_sparse'
    LK_SPARSEMASK = 'lk_sparsemask'
    LK_DENSE = 'lk_dense'
    DTVL1_DENSE = 'dtvl1_dense'
    SF_DENSE = 'sf_dense'
    RLOF_DENSE = 'rlof_dense'

@dataclass
class TimeInterval:
    start: datetime
    stop: datetime