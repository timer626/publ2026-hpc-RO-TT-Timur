from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import logging

logger = logging.getLogger(__name__)

class DataEmbeddings:
    def __init__(self,settings):
        self.embedding = settings.embedding
        self.dataset = settings.dataset_raw
        self.constants = settings.constants
    def load_embedding(self):
        local_dir = self.embedding.local_dir
        padding_side = self.constants.padding_side
        tokenizer = AutoTokenizer.from_pretrained(local_dir,local_files_only=True, padding_side=padding_side)
        model = AutoModel.from_pretrained(local_dir,local_files_only=True)
        return {'tokenizer':tokenizer,'model':model}
    def load_dataset(self):
        local_dir = self.dataset.local_dir
        dataset = load_from_disk(local_dir)
        return dataset
    
    def load(self):
        embedding = self.load_embedding()
        dataset = self.load_dataset()
        return {'tokenizer':embedding.get('tokenizer'),'model':embedding.get('model'),'dataset':dataset}

def prepare_data_embeddings(settings):
    data_embeddings = DataEmbeddings(settings)
    return data_embeddings.load()
        
        
if __name__=="__main__":
    pass
