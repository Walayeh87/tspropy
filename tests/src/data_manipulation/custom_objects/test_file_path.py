from pathlib import Path

import pytest

from src.data_manipulation.custom_objects.file_path import FilePath

TESTING_FILE_PATH = r"tests/data/files_for_extension_checking"


def test_stripping() -> None:
    file_path = FilePath(path=rf"{TESTING_FILE_PATH}/file_without_extension ")

    assert file_path.path == Path(r"tests/data/files_for_extension_checking/file_without_extension")


def test_against_non_existing_file() -> None:
    with pytest.raises(FileNotFoundError):
        FilePath(path=rf"{TESTING_FILE_PATH}/not_exist_file")
