import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_from_disk
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelTransformers:
    def __init__(self,local_files:Path,settings):
        self.local_files_path = local_files
        self.settings = settings
    def __call__(self,):
        tokenizer = AutoTokenizer.from_pretrained(self.local_files_path,local_files_only=True,padding_side=self.settings.constants.padding_side)
        model = AutoModelForCausalLM.from_pretrained(self.local_files_path,local_files_only=True)
        return {'tokenizer':tokenizer, 'model':model}
class ModelEmbedder:
    def __init__(self,local_files:Path,settings):
            self.local_files_path = local_files
            self.settings = settings
    def __call__(self,):
        tokenizer = AutoTokenizer.from_pretrained(self.local_files_path,local_files_only=True,padding_side=self.settings.constants.padding_side)
        model = AutoModel.from_pretrained(self.local_files_path,local_files_only=True)
        return {'tokenizer':tokenizer, 'model':model}
    
class DatasetHF:
    def __init__(self,local_files:Path):
        self.local_files_path = local_files
    def __call__(self):
        dataset = load_from_disk(self.local_files_path)
        return dataset
    
        
def load_from_disk(settings):
    models = settings.models
    datasets = settings.datasets
    models_loaded = {}
    datasets_loaded = {}
    for model in models:
        local_dir = model.local_dir
        model_name = model.model_name
        model_loaded = ModelTransformers(local_dir,settings)
        models_loaded[model_name] = model_loaded()
        logging.info(f"{local_dir} loaded to memory")
    logger.info('All models loaded to memory')
    for dataset in datasets:
        local_dir = dataset.local_dir
        dataset_name = dataset.dataset_name
        dataset_loaded =  DatasetHF(local_dir)
        datasets_loaded[dataset_name] = dataset_loaded()
        logging.info(f"{local_dir} loaded to memory")
    logger.info('All datasets loaded to memory')
    return models_loaded,datasets_loaded
        
if __name__=="__main__":
    pass