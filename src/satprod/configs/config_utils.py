import yaml
from munch import Munch
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
import os

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
    GRID = 'grid'

@dataclass
class TimeInterval:
    start: datetime
    stop: datetime
    
def structurize_wind_grid_images():
    cd = str(os.path.dirname(os.path.abspath(__file__)))
    root = f'{cd}/../../..'
    grid_path = f'{root}/data/img/grid'
    images = os.listdir(grid_path)
    for image in images:
        if 'T' not in image:
            continue
        T_split = image.split('T')
        
        datelist = T_split[0].split('-')
        year = datelist[0]
        month = datelist[1]
        day = datelist[2]
        
        newname = T_split[1]
        os.makedirs(os.path.join(grid_path, year, month, day), exist_ok=True)
        os.rename(os.path.join(grid_path, image), os.path.join(grid_path, year, month, day, newname))