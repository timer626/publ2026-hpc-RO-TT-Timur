from setup.create_project_structure import create_project_structure
from setup.download_data import download_data
from setup.load_from_disk import load_from_disk
from setup.create_embeddings import create_embeddings
from config import get_settings
from logging_config import setup_logging
from dotenv import load_dotenv
from pathlib import Path


class Setup:
    def __init__(self,project_root:Path):
        self.settings = get_settings(project_root)
        load_dotenv()
    def prepare_project_logging(self):
        setup_logging()
    def prepare_project_structure(self):
        create_project_structure(self.settings)
    def prepare_project_data(self):
        download_data(self.settings)
        models_loaded, datasets_loaded = load_from_disk(self.settings)
        create_embeddings(models_loaded,datasets_loaded,self.settings)
    def prepare_project(self):
        self.prepare_project_structure()
        self.prepare_project_logging()
        self.prepare_project_data()


def prepare():
    project_root = Path(__name__).resolve().parent
    setup = Setup(project_root)
    setup.prepare_project()
    
    
    

def main():
    prepare()
    

if __name__ == "__main__":
    main()
