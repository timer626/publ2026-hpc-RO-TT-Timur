from create_init_files import create_init_files
from download_data import download_data
from load_from_disk import load_from_disk
from create_embeddings import create_embeddings
from config import settings
from dotenv import load_dotenv
import os

load_dotenv()



def main():
    hf_token = os.getenv("HF_TOKEN")
    data = create_init_files()
    download_data(data,settings)
    models_loaded, datasets_loaded = load_from_disk()
    create_embeddings(models_loaded,datasets_loaded,settings)

if __name__ == "__main__":
    main()
