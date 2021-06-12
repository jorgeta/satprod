from satprod.pipelines.evaluation import Evaluate
import numpy as np
import pandas as pd
import os

class Comparison():
    
    def __init__(self, park: str, model_ids: dict=None):
        
        self.park = park
        
        cd = str(os.path.dirname(os.path.abspath(__file__)))
        self.root = f'{cd}/../../..'
        self.TE_data_path = f'{self.root}/data/prod_forecasts'
        
        if model_ids is None:
            model_ids = { # 'chosen_name' : ['model_name', 'timestamp']
                'LSTM': ['LSTM', '2021-06-06-17-22'],
                #'LSTM_img': ['LSTM', '2021-06-10-01-23'],
                'MLR': ['MLR', '2021-06-11-05-21'],
                #'TCN': ['TCN', '2021-06-10-09-12'],
                'TCN_img': None,
                'TCN_Bai': ['TCN_Bai', '2021-06-10-20-08'],
                'TCN_Bai_img': None
            }
        
        eval_dict = {}
        for key, value in model_ids.items():
            if value is not None:
                eval_dict[key] = Evaluate(value[1], value[0], self.park, 
                                        save_info_file=False,
                                        run_persistence=False, 
                                        save_error_plots=False,
                                        save_fitting_examples=False)
                
        self.load_TE_predicions()
            
    def load_TE_predicions(self):
        TE_df = pd.read_csv(f'{self.TE_data_path}/{self.park}_prod_forecasts.csv')
        print(TE_df)
        
if __name__=='__main__':
    comp_skom = Comparison('skom')