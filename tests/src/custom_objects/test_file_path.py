import pytest

from src.custom_objects.file_path import FilePath


def test_against_non_existing_file() -> None:
    with pytest.raises(FileNotFoundError):
        FilePath(path="not_exist_file")


# def test_existing_file() -> None:
#     path = "../tests/data/file_without_extension"
#     file_path = FilePath(path=path)
#
#     assert file_path.path == Path(path)
