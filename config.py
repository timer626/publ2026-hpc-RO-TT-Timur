import logging
import tomllib
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class GeneratorConfig(BaseModel):
    repo_id: str
    model_name: str
    revision: str
    local_dir: Path

class DatasetRawConfig(BaseModel):
    repo_id: str
    dataset_name:str
    revision: str
    name: str
    split: str
    local_dir: Path
class DatasetPreparedConfig(BaseModel):
    local_dir: Path
    local_file_documents: Path
    local_index_file_documents: Path
    local_file_embeddings: Path
    
class EmbeddingConfig(BaseModel):
    repo_id: str
    model_name: str
    revision: str
    local_dir: Path 

class LogConfig(BaseModel):
    local_dir: Path
 
class ConstantConfig(BaseModel):
    dev: bool
    device_cuda: bool
    padding_side: str
    max_length: int
    batch_size: int
    padding: bool
    truncation: bool
    is_first_run: bool

class Settings(BaseModel):
    generator: GeneratorConfig
    dataset_raw: DatasetRawConfig
    dataset_prepared: DatasetPreparedConfig
    embedding: EmbeddingConfig
    logs: LogConfig
    constants: ConstantConfig

def resolve_path(path: Path, project_root: Path) -> Path:
    return (project_root / path).resolve()


def get_settings(project_root: Path) -> Settings:
    config_path = project_root / "configs" / "config.toml"
    config_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    settings = Settings(**config_data)
    settings.generator.local_dir = resolve_path(settings.generator.local_dir, project_root)
    settings.dataset_raw.local_dir = resolve_path(settings.dataset_raw.local_dir, project_root)
    settings.dataset_prepared.local_dir = resolve_path(settings.dataset_prepared.local_dir, project_root)
    settings.dataset_prepared.local_file_documents = resolve_path(settings.dataset_prepared.local_file_documents, project_root)
    settings.dataset_prepared.local_index_file_documents = resolve_path(settings.dataset_prepared.local_index_file_documents, project_root)
    settings.dataset_prepared.local_file_embeddings = resolve_path(settings.dataset_prepared.local_file_embeddings, project_root)
    settings.embedding.local_dir = resolve_path(settings.embedding.local_dir, project_root)
    settings.logs.local_dir = resolve_path(settings.logs.local_dir, project_root)
    
    logger.info(f"Настройки из {str(config_path)} загружены")

    return settings
if __name__=="__main__":
    project_root = Path(__file__).resolve().parent
    get_settings(project_root)
