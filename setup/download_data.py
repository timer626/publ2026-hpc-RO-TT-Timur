from huggingface_hub import snapshot_download
from datasets import load_dataset

import logging

logger = logging.getLogger(__name__)

class DownloadData:
    def __init__(self,settings):
        self.generator = settings.generator
        self.embedding = settings.embedding
        self.dataset = settings.dataset_raw
    def download_generator(self):
        repo_id = self.generator.repo_id
        revision = self.generator.revision
        local_dir = self.generator.local_dir
        snapshot_download(repo_id=repo_id, revision=revision,local_dir=local_dir)
        logger.info(f'Have downloaded {repo_id} to {local_dir}')
    def download_embedding(self):
        repo_id = self.embedding.repo_id
        revision = self.embedding.revision
        local_dir = self.embedding.local_dir
        snapshot_download(repo_id=repo_id, revision=revision,local_dir=local_dir)
        logger.info(f'Have downloaded {repo_id} to {local_dir}')
        
    def download_dataset(self):
        repo_id = self.dataset.repo_id
        revision = self.dataset.revision
        name = self.dataset.name
        split = self.dataset.split
        local_dir = self.dataset.local_dir
        dataset = load_dataset(path=repo_id,name=name,revision=revision,split=split)
        dataset.save_to_disk(local_dir)
        logger.info(f'Have downloaded {repo_id} {name}  to {local_dir}')
    def download(self):
        self.download_generator()
        self.download_embedding()
        self.download_dataset()
        logger.info(f"Скачаны все данные")
        
        
def download_data(settings):
    DownloadData(settings).download()
    
if __name__=="__main__":
    pass
        
        
        
    
