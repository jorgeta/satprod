import fire
import os
from satprod.configs.config_utils import read_yaml

class App:

    def __init__(self):
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.config = read_yaml(f'{cd}/../../config.yaml')

def main():
    fire.Fire(App)

if __name__ == '__main__':
    main()