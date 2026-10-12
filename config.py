import tomllib
from pathlib import Path

from pydantic import BaseModel

class ModelConfig(BaseModel):
    repo_id: str
    model_name: str
    revision: str
    local_dir: str


class DatasetConfig(BaseModel):
    repo_id: str
    dataset_name:str
    revision: str
    name: str
    split: str
    local_dir: str

class LogConfig(BaseModel):
    local_dir: str
class EmbeddingConfig(BaseModel):
    model_name: str
    local_dir:str
    local_file_documents:str
    local_index_file_documents:str
    local_file_embeddings:str
    
    
class ConstantConfig(BaseModel):
    dev: bool
    device_cuda: bool
    padding_side: str
    max_length: int
    batch_size: int
    padding: bool
    truncation: bool
    marker_file: str

class Settings(BaseModel):
    models: list[ModelConfig]
    datasets: list[DatasetConfig]
    logs: LogConfig
    embeddings: EmbeddingConfig
    constants: ConstantConfig

config_data = tomllib.loads(Path(r"./config.toml").read_text(encoding="utf-8"))
settings = Settings(**config_data)