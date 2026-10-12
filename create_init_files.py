from config import settings

from pathlib import Path
import json

import logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

class Dirs:
    def __init__(self,settings):
        models = settings.models
        datasets = settings.datasets
        embeddings = settings.embeddings
        log_dir = settings.logs.local_dir
        dirs = set()
        for model in models:
            dirs.add(Path(model.local_dir))
        for dataset in datasets:
            dirs.add(Path(dataset.local_dir))
        dirs.add(Path(embeddings.local_dir))
        dirs.add(Path(log_dir))
        for dir in dirs:
            dir.mkdir(parents=True,exist_ok=True)
        logging.info(f"Созданы директории {', '.join([dir.name for dir in dirs])} ")
class InitFiles:
    def __init__(self,settings):
        self.models = settings.models
        self.datasets = settings.datasets
        self.marker_file = Path(settings.constants.marker_file)
    def create_marker_file(self,):
        info=[]
        for model in self.models:
            info.append({'repo_id':model.repo_id,
                'revision':model.revision,
                'is_downloaded':False})
        for dataset in self.datasets:
            info.append({'repo_id':dataset.repo_id,
                        'revision':dataset.revision,
                        'is_downloaded':False})
        
        with open(self.marker_file,'w',encoding='UTF-8') as f:
            json.dump(info,f,ensure_ascii=False,indent=4)
        logging.info(f"Создан файл {self.marker_file}")
 
class CheckerFiles:
    def __init__(self,init_files):
        self.init_files = init_files
        self.marker_file_path = init_files.marker_file
    def marker_file(self):
        if not self.marker_file_path.exists():
            self.init_files.create_marker_file()
    def not_downloaded_data(self,):
        repos_id = []
        revisions = []
        if self.marker_file_path.exists():
            with open(self.marker_file_path,'r',encoding='UTF-8') as f:
                marker_file = json.load(f)
            for item in marker_file:
                if item['is_downloaded']==False:
                    repos_id.append(item['repo_id'])
                    revisions.append(item['revision'])
                
        return {'repos_id':repos_id, 'revisions':revisions}
        
def create_init_files():
    init_files = InitFiles(settings)
    Dirs(settings)
    checker_files = CheckerFiles(init_files)
    checker_files.marker_file()
    data = checker_files.not_downloaded_data()
    if (data.get('repos_id',[]) and data.get('revisions',[])):
        logger.info(f'не хватает файлов {data}')
        return data
    else:
        logger.info('Все файлы есть')
        return {}
    
     
 
if __name__=="__main__":
    create_init_files()