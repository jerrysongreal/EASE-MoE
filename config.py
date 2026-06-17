"""
EASE-MoE Configuration File
"""

import os

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    
    ROBERTA_PATH = "roberta-base"
    SWIN_PATH = "microsoft/swin-base-patch4-window7-224"
    
    SEED = 42
    DEVICE = 0
    BATCH_SIZE = 32
    
    HIDDEN_DIM = 512
    LLM_EMB_DIM = 768
    
    NUM_CHUNKS = 4
    NUM_HEADS = 8
    DROPOUT = 0.1
    
    LR = 5e-5
    WEIGHT_DECAY = 5e-5
    
    ALPHA = 0.2
    BETA = 0.5
    GAMMA = 0.3
    DELTA = 0.1
    
    EPOCHS_STAGE1 = 50
    EPOCHS_STAGE2 = 100
    N_FOLDS = 5
    LABEL_RATIO = 0.3
    PSEUDO_THRESHOLD = 0.95
    
    MAX_TEXT_LENGTH = 512
    MAX_COMMENT_LENGTH = 128
    MAX_COMMENTS = 15
    
    @classmethod
    def ensure_dirs(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)


class DatasetConfig:
    GOSSIPCOP = {
        'name': 'gossipcop',
        'json_file': 'gossipcop/gossipcop.json',
        'image_folder': 'gossipcop/images',
        'empathy_csv': 'gossipcop/empathy.csv'
    }
    
    POLITIFACT = {
        'name': 'politifact',
        'json_file': 'politifact/politifact.json',
        'image_folder': 'politifact/images',
        'empathy_csv': 'politifact/empathy.csv'
    }
    
    PHEME = {
        'name': 'pheme',
        'json_file': 'pheme/pheme.json',
        'image_folder': 'pheme/images',
        'empathy_csv': 'pheme/empathy.csv'
    }
    
    WEIBO = {
        'name': 'weibo',
        'json_file': 'weibo/weibo.json',
        'image_folder': 'weibo/images',
        'empathy_csv': 'weibo/empathy.csv'
    }


class TrainingConfig:
    STAGE1 = {
        'epochs': 50,
        'lr': 5e-5,
        'weight_decay': 5e-5,
        'scheduler_step': 15,
        'scheduler_gamma': 0.5
    }
    
    STAGE2 = {
        'epochs': 100,
        'lr': 5e-6,
        'weight_decay': 2.5e-5,
        'scheduler_step': 10,
        'scheduler_gamma': 0.5
    }


if __name__ == "__main__":
    Config.ensure_dirs()
    print(f"Project root: {Config.PROJECT_ROOT}")
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Model directory: {Config.MODEL_DIR}")
    print(f"Log directory: {Config.LOG_DIR}")
