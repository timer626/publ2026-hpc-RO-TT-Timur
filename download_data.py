from huggingface_hub import snapshot_download
import tomllib
from pathlib import Path
from datasets import load_dataset

import logging


logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)
def main():
    config_path = Path(r'./config.toml')
    with open(config_path,'rb') as f:
        config = tomllib.load(f)
    models = config['models']
    datasets = config['datasets']
    for model in models:
        repo_id = model['repo_id']
        revision = model['revision']
        local_dir = model['local_dir']
        snapshot_download(repo_id=repo_id, revision=revision,local_dir=local_dir)
        logging.info(f'Have downloaded {repo_id} to {local_dir}')
        logging.info('all models have downloaded')
    for dataset in datasets:
        repo_id = dataset['repo_id']
        revision = dataset['revision']
        name = dataset["name"]
        split = dataset['split']
        local_dir = dataset['local_dir']
        dataset = load_dataset(path=repo_id,name=name,revision=revision,split=split)
        dataset.save_to_disk(local_dir)
        logging.info(f'Have downloaded {repo_id} {name}  to {local_dir}')
            
    logging.info('all datasets downloaded')
        
    logging.info("all data have downloaded")
    
    

if __name__=="__main__":
    main()
    