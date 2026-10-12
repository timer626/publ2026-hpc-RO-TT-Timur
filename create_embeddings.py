from config import settings
import logging

import torch
import torch.nn.functional as F

from torch import Tensor
from load_from_disk import load_data_from_disk
from pathlib import Path
import json

from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

class Documents:
    def __init__(self,models_loaded,datasets_loaded,settings):
        self.output_file_documents = Path(settings.embeddings.local_file_documents)
        self.output_index_file_documents = Path(settings.embeddings.local_index_file_documents)
        self.output_file_embeddings = Path(settings.constants.local_file_embeddings)
        self.dataset = datasets_loaded.get(settings.datasets.dataset_name,None)
        embedder = models_loaded.get(settings.embeddings.model_name,None)
        self.tokenizer,self.model = embedder['tokenizer'], embedder['model']
        
        self.device = 'cuda' if (settings.constants.device_cuda and torch.cuda.is_available()) else 'cpu'
        self.max_length = settings.constants.max_length
        self.batch_size = settings.constants.batch_size
        self.padding = settings.constants.padding
        self.truncation = settings.constants.truncation
        
        
    def make_documents(self):
        self.documents = []
        for example in self.dataset:
            example_id = example["id"]
            titles = example["context"]["title"]
            sentences_list = example["context"]["sentences"]
    
            for title, sentences in zip(titles, sentences_list):
                text = (f"Title: {title}\n"f"Text: {' '.join(sentences)}")
                self.documents.append({"text": text,"title": title,"example_id": example_id})
        with open(self.output_file_documents,'w',encoding='UTF-8') as f:
            for doc in self.documents:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        logging.info(f'Документы сохранены в {str(self.output_file_documents)} ')

    def build_document_index(self) -> dict[str, list[int]]:
        index = defaultdict(list)
        for i, doc in enumerate(self.documents):
            index[doc["example_id"]].append(i)
        with open(self.output_index_file_documents,'w',encoding='UTF-8') as f:
            json.dump(dict(index),f,ensure_ascii=False)
        logging.info(f'Индекс для документов сохранен в {str(self.output_index_file_documents)} ')
    
    def last_token_pool(self,last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def make_embeddings(self):
        
        self.model = self.model.to(self.device)
        self.model.eval()
        all_embeddings = []
        for start in range(0,len(self.documents),self.batch_size):
            batch_documents = self.documents[start:start + self.batch_size]
            texts = [doc["text"] for doc in batch_documents]
            inputs = self.tokenizer(texts,
                                    padding=self.padding,
                                    truncation=self.truncation,
                                    max_length=self.max_length,
                                    return_tensors="pt",).to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                embeddings = self.last_token_pool(outputs.last_hidden_state,inputs["attention_mask"])
                embeddings = F.normalize(embeddings.float(),p=2,dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
        embeddings = np.concatenate(all_embeddings, axis=0)
        np.save(self.output_file_embeddings, embeddings)
        logging.info(f"Эмбеддинги документов сохранены в {str(self.output_file_embeddings)}")
    
    def prepare(self):
        self.make_documents()
        self.build_document_index()
        self.make_embeddings()
        logging.info(f"Документы подготовлены для генерации")
        
                
def create_embeddings(models_loaded,datasets_loaded,settings):
    docs = Documents(models_loaded,datasets_loaded,settings)
    docs.prepare()
 
    

if __name__=="__main__":
    if settings.constants.dev:
        logging.info("Приложение работает в dev режиме")
    else:
        create_embeddings()
        