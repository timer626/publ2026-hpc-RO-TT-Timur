from config import settings

from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

class Dirs:
    def __init__(self,settings):
        models = settings.models
        datasets = settings.datasets
        log_dir = settings.logs.local_dir
        dirs = set()
        for model in models:
            dirs.add(Path(model.local_dir))
        for dataset in datasets:
            dirs.add(Path(dataset.local_dir))
        dirs.add(Path(log_dir))
        for dir in dirs:
            dir.mkdir(parents=True,exist_ok=True)
        logging.info(f"Созданы директории {', '.join([dir.name for dir in dirs])} ")
if __name__=="__main__":
    Dirs(settings)
        
        
            
            