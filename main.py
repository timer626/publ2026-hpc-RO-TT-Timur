from pathlib import Path

from config import get_settings
from setup.setup import Setup


def prepare():
    project_root = Path(__file__).resolve().parent
    settings = get_settings(project_root)
    setup = Setup(settings)
    setup.prepare_project()


def main():
    prepare()

if __name__ == "__main__":
    main()
