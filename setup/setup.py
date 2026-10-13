import logging

from dotenv import load_dotenv

from logging_config import setup_logging
from setup.create_project_structure import create_project_structure
from setup.download_data import download_data
from setup.prepare_data_embeddings import prepare_data_embeddings
from setup.create_embeddings import create_embeddings


logger = logging.getLogger(__name__)


class Setup:
    def __init__(self, settings):
        self.settings = settings
        load_dotenv()

    def prepare_project_logging(self):
        setup_logging()

    def prepare_project_structure(self):
        create_project_structure(self.settings)

    def prepare_project_data(self):
        download_data(self.settings)
        data_embeddings = prepare_data_embeddings(self.settings)
        create_embeddings(data_embeddings, self.settings)

    def prepare_project(self):
        self.prepare_project_logging()

        if not self.settings.constants.is_first_run:
            logger.info("Первичная подготовка завершена; проект готов к запуску экспериментов")
            return

        logger.info("Первый запуск: выполняется подготовка данных и базы эмбеддингов")
        logger.info("Подготовка структуры проекта")
        self.prepare_project_structure()
        logger.info("Подготовка данных проекта")
        self.prepare_project_data()

        logger.info(
            "Первичная подготовка завершена. После проверки файлов установите "
            "constants.is_first_run = false в config.toml"
        )
