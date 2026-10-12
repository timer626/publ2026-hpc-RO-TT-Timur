from huggingface_hub import snapshot_download
from datasets import load_dataset
from config import settings
import json



import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

class DownloadData:
    def __init__(self,data,settings):
        self.marker_file = settings.constants.marker_file
        models = settings.models
        datasets = settings.datasets
        repos_id = data['repos_id']
        revisions = data['revisions']
        for model in models:
            repo_id = model.repo_id
            revision = model.revision
            if (repo_id in repos_id and revision in revisions):
                local_dir = model.local_dir
                snapshot_download(repo_id=repo_id, revision=revision,local_dir=local_dir)
            self.update_download_status(repo_id,revision)
            logging.info(f'Have downloaded {repo_id} to {local_dir}')
        for dataset in datasets:
            repo_id = dataset.repo_id
            revision = dataset.revision
            if (repo_id in repos_id and revision in revisions):
                name = dataset.name
                split = dataset.split
                local_dir = dataset.local_dir
                dataset = load_dataset(path=repo_id,name=name,revision=revision,split=split)
                dataset.save_to_disk(local_dir)
                self.update_download_status(repo_id,revision)
                logging.info(f'Have downloaded {repo_id} {name}  to {local_dir}')
        logging.info('all models have downloaded')
        
    def update_download_status(self,repo_id: str,revision: str,) -> None:
        with open(self.marker_file,"r", encoding="utf-8") as f:
            repositories = json.load(f)
        for item in repositories:
            if (item["repo_id"] == repo_id and item["revision"] == revision):
                item["is_downloaded"] = True
                break
        with open(self.marker_file,"w", encoding="utf-8") as f:
            json.dump(repositories, f, ensure_ascii=False, indent=4)

        logger.info("Статус обновлён: %s, downloaded=%s",repo_id,True,)
        
def download_data(data,settings):
    DownloadData(data,settings)
    
if __name__=="__main__":
    logging.info(f"{settings.constants}")
    if settings.constants.dev:
        logging.info(f"Приложение в dev режиме")
    else:
        DownloadData()
        
        
        
    