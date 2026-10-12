from huggingface_hub import snapshot_download
from datasets import load_dataset
from config import settings

import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
def download_data_from_hf():
    models = settings.models
    datasets = settings.datasets
    for model in models:
        repo_id = model.repo_id
        revision = model.revision
        local_dir = model.local_dir
        snapshot_download(repo_id=repo_id, revision=revision,local_dir=local_dir)
        logging.info(f'Have downloaded {repo_id} to {local_dir}')
        logging.info('all models have downloaded')
    for dataset in datasets:
        repo_id = dataset.repo_id
        revision = dataset.revision
        name = dataset.name
        split = dataset.split
        local_dir = dataset.local_dir
        dataset = load_dataset(path=repo_id,name=name,revision=revision,split=split)
        dataset.save_to_disk(local_dir)
        logging.info(f'Have downloaded {repo_id} {name}  to {local_dir}')
            
    logging.info('all datasets downloaded')
        
    logging.info("all data have downloaded")
    
    

if __name__=="__main__":
    logging.info(f"{settings.constants}")
    if settings.constants.dev:
        logging.info(f"Приложение в dev режиме")
    else:
        download_data_from_hf()
        
    