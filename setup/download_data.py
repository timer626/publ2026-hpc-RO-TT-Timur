from huggingface_hub import snapshot_download
from datasets import load_dataset

import logging

logger = logging.getLogger(__name__)

class DownloadData:
    def __init__(self,settings):
        self.marker_file = settings.constants.local_marker_file
        models = settings.models
        datasets = settings.datasets
        for model in models:
            repo_id = model.repo_id
            revision = model.revision
            local_dir = model.local_dir
            snapshot_download(repo_id=repo_id, revision=revision,local_dir=local_dir)
            logger.info(f'Have downloaded {repo_id} to {local_dir}')
        for dataset in datasets:
            repo_id = dataset.repo_id
            revision = dataset.revision
            name = dataset.name
            split = dataset.split
            local_dir = dataset.local_dir
            dataset = load_dataset(path=repo_id,name=name,revision=revision,split=split)
            dataset.save_to_disk(local_dir)
            logging.info(f'Have downloaded {repo_id} {name}  to {local_dir}')
        logger.info('all models have downloaded')
        
        
def download_data(settings):
    DownloadData(settings)
    
if __name__=="__main__":
    pass
        
        
        
    