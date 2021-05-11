from dataclasses import dataclass
from datetime import datetime

@dataclass
class TrainConfig():
    learning_rate: int
    scheduler_step_size: int
    scheduler_gamma: float
    train_valid_splits: int
    pred_sequence_length: int
    parks: [str]
    num_feature_types: [str]
    img_features: [str]
    img_extraction_method: str
    batch_size: int = 64
    num_epochs: int = 150
    valid_start: datetime = datetime(2018, 6, 1, 0)
    test_start: datetime = datetime(2019, 1, 1, 0)
    random_seed: int = 0
    
    def __post__init__(self):
        assert len(img_features)==1, "Only one image can be used at a time."
        assert len(parks)==1 or len(parks)==4, "Choose either one or all parks."