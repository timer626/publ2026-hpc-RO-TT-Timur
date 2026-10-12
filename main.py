from create_init_files import create_init_files
from download_data import download_data
from config import settings
from dotenv import load_dotenv
import os

load_dotenv()



def main():
    #print("Hello from tt-kv-compress!")
    hf_token = os.getenv("HF_TOKEN")
    data = create_init_files()
    download_data(data,settings)
    


if __name__ == "__main__":
    main()
