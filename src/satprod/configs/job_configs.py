from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainConfig():
    model: str
    learning_rate: int
    pred_sequence_length: int
    parks: [str]
    num_feature_types: [str]
    img_features: [str]
    img_extraction_method: str
    scheduler_gamma: float = 0.5
    train_on_small_subset: bool = False
    use_numerical_forecasts: bool = True
    use_img_forecasts: bool = False
    batch_size: int = 64
    num_epochs: int = 150
    valid_start: datetime = datetime(2019, 5, 12, 0)
    test_start: datetime = datetime(2020, 5, 12, 0)
    random_seed: int = 0
    recursive: bool = False
    
    def __post_init__(self):
        assert len(self.img_features)<=1, "Not more than one image can be used at a time."
        assert len(self.parks)==1 or len(self.parks)==4, "Choose either one or all parks."
        
        self.scheduler_step_size = int(0.9*self.num_epochs)
        
        if self.model=='TCN':
            self.recursive = True
            self.use_numerical_forecasts = True
            self.use_img_forecasts = True
        else:
            self.use_img_forecasts = False
        if len(self.img_features)==0:
            self.use_img_forecasts = False