from satprod.pipelines.evaluation import ModelEvaluation
import numpy as np
import pandas as pd
import os

class ModelComparison():
    
    def __init__(self, park: str, config):
        
        self.park = park
        self.config = config[park]
        
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        self.TE_data_path = f'{self.root}/data/prod_forecasts'
        
        '''print(self.config)
        
        eval_dict = {}
        for key, value in self.config.items():
            if value is None: continue
            value['park']=self.park
            eval_dict[key] = ModelEvaluation(**value)'''
        
        self.load_TE_predicions()
            
    def load_TE_predicions(self):
        TE_df = pd.read_csv(f'{self.TE_data_path}/{self.park}_prod_forecasts.csv')
        print(TE_df)
        