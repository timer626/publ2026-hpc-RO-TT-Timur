import logging


logger = logging.getLogger(__name__)


class InitFiles:
    def __init__(self, settings):
        self.settings = settings

    def create_dirs(self):
        directories = (
            self.settings.generator.local_dir,
            self.settings.dataset_raw.local_dir,
            self.settings.dataset_prepared.local_dir,
            self.settings.embedding.local_dir,
            self.settings.logs.local_dir,
        )

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        created_directories = ", ".join(str(directory) for directory in directories)
        logger.info("Подготовлены директории: %s", created_directories)

    def create_project_structure(self):
        self.create_dirs()
        logger.info("Структура проекта создана")


def create_project_structure(settings):
    init_files = InitFiles(settings)
    init_files.create_project_structure()


if __name__ == "__main__":
    pass
