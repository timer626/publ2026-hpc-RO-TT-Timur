import json
import logging
logger = logging.getLogger(__name__)

class InitFiles:
    def __init__(self,settings):
        self.models = settings.models
        self.datasets = settings.datasets
        self.embeddings = settings.embeddings
        self.logs = settings.logs
        self.marker_file = settings.constants.local_marker_file
    def create_dirs(self):
        dirs = set()
        for model in self.models:
            dirs.add(model.local_dir)
        for dataset in self.datasets:
            dirs.add(dataset.local_dir)
        dirs.add(self.embeddings.local_dir)
        dirs.add(self.logs.local_dir)
        for dir in dirs:
            dir.mkdir(parents=True,exist_ok=True)
        logger.info(f"Созданы директории {', '.join([dir.name for dir in dirs])}")
        
    def create_marker_file(self):
        info=[]
        for model in self.models:
            info.append({'repo_id':model.repo_id,'revision':model.revision,'is_downloaded':False})
        for dataset in self.datasets:
            info.append({'repo_id':dataset.repo_id,'revision':dataset.revision,'is_downloaded':False})
        with open(self.marker_file,'w',encoding='UTF-8') as f:
            json.dump(info,f,ensure_ascii=False,indent=4)
        logger.info(f"Создан файл {self.marker_file}")
    def create_project_structure(self):
        self.create_dirs()
        self.create_marker_file()
        logger.info("Структура проекта создана")
 
        
def create_project_structure(settings):
    init_files = InitFiles(settings)
    init_files.create_project_structure()
    
 
if __name__=="__main__":
    pass