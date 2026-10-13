import tomllib
from pathlib import Path
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

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
    local_marker_file: str

class Settings(BaseModel):
    models: list[ModelConfig]
    datasets: list[DatasetConfig]
    logs: LogConfig
    embeddings: EmbeddingConfig
    constants: ConstantConfig

def resolve_path(path:str,project_root_absolute:Path):
    path = Path(path)
    path = project_root_absolute / path
    return path.resolve()


def get_settings(project_root:Path):
    config_path = project_root/Path(r'./configs/config.toml')
    config_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    settings = Settings(**config_data)
    for model in settings.models:
        model.local_dir = resolve_path(model.local_dir,project_root)
    for dataset in settings.datasets:
        dataset.local_dir = resolve_path(dataset.local_dir,project_root)
    settings.logs.local_dir = resolve_path(settings.logs.local_dir, project_root)
    settings.embeddings.local_dir = resolve_path(settings.embeddings.local_dir, project_root)
    settings.embeddings.local_file_documents = resolve_path(settings.embeddings.local_file_documents, project_root)
    settings.embeddings.local_index_file_documents = resolve_path(settings.embeddings.local_index_file_documents, project_root)
    settings.embeddings.local_file_embeddings = resolve_path(settings.embeddings.local_file_embeddings, project_root)
    settings.constants.local_marker_file = resolve_path(settings.constants.local_marker_file, project_root)
    logger.info(f"Настройки из {str(config_path)} загружены")

    return settings