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
    DENSE = 'dense'
    SPARSE = 'sparse'
    SPARSEMASK = 'sparsemask'

@dataclass
class TimeInterval:
    start: datetime
    stop: datetime