from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainConfig():
    """Parameters directly related to the training of the model,
    independent of the dataset and the specific model that is chosen.
    """
    
    learning_rate: int = 4.0e-3
    scheduler_gamma: float = 0.5
    train_on_one_batch: bool = False
    batch_size: int = 64
    num_epochs: int = 150
    random_seed: int = 0
    weight_decay: float = 0.0
    predict_current_time: bool = False
    
    def __post_init__(self):
        # lower the learning rate with a factor 0.9 after 90 % of the epochs has passed
        self.scheduler_step_size = int(0.9*self.num_epochs)
        if self.scheduler_step_size < 1:
            self.scheduler_step_size = 1
        else:
            self.scheduler_step_size = 5

@dataclass
class DataConfig():
    """Information about features and splitting of data for
    building a dataset that fits the model structure and vice versa.
    """
    
    model: str
    parks: [str]
    numerical_features: [str]
    use_img_features: bool
    img_extraction_method: str
    
    valid_start: dict #datetime = datetime(2019, 5, 12, 0)
    test_start: dict #datetime = datetime(2020, 5, 12, 0)
    test_end: dict #datetime = datetime(2021, 5, 11, 23)
    
    pred_sequence_length: int = 5
    use_numerical_forecasts: bool = True
    use_img_forecasts: bool = False
    crop_image: bool = False
    
    def __post_init__(self):
        self.img_features = ['grid'] if self.use_img_features else []
        
        assert len(set(self.parks))==len(self.parks), 'The list of parks contains duplicates'
        assert len(self.parks)==1 or len(self.parks)==4, "Choose either one or all parks."
        
        self.valid_start = datetime(**self.valid_start)
        self.test_start = datetime(**self.test_start)
        self.test_end = datetime(**self.test_end)
        
        if self.use_img_features and self.model=='TCN':
            self.use_img_forecasts = True
        
        if not self.use_img_features: 
            self.img_extraction_method = None
            self.use_img_forecasts = False
            
        if len(self.numerical_features) < 1:
            self.use_numerical_forecasts = False