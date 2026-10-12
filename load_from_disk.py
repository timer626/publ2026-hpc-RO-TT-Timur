import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from config import settings
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

class ModelTransformers:
    def __init__(self,local_files:str):
        self.local_files_path = Path(local_files)
    def __call__(self,):
        tokenizer = AutoTokenizer(self.local_files_path,local_files_only=True,padding_side=settings.constants.padding_side)
        model = AutoModelForCausalLM(self.local_files_path,local_files_only=True)
        return {'tokenizer':tokenizer, 'model':model}
class DatasetHF:
    def __init__(self,local_files:str):
        self.local_files_path = Path(local_files)
    def __call__(self):
        dataset = load_from_disk(self.local_files_path)
        return dataset
    
        
def load_from_disk():
    models = settings.models
    datasets = settings.datasets
    models_loaded = {}
    datasets_loaded = {}
    for model in models:
        local_dir = model.local_dir
        model_name = model.model_name
        models_loaded[model_name] = ModelTransformers(local_dir)
        logging.info(f"{local_dir} loaded to memory")
    logging.info('All models loaded to memory')
    for dataset in datasets:
        local_dir = dataset.local_dir
        dataset_name = dataset.dataset_name
        datasets_loaded[dataset_name] = DatasetHF(local_dir)
        logging.info(f"{local_dir} loaded to memory")
    logging.info('All datasets loaded to memory')
    return models_loaded,datasets_loaded
        

if __name__=="__main__":
    logging.info(settings.constants.dev)
    if settings.constants.dev:
        logging.info("Приложение в dev режиме")
    else:
        models_loaded,datasets_loaded = load_from_disk()