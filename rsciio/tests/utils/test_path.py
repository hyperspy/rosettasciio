from pathlib import Path

from rsciio.utils.path import (
    append2pathname,
    ensure_directory,
    incremental_filename,
    overwrite,
)


def test_append2pathname():
    filename = "data/file.txt"
    to_append = "_std"
    new_filename = append2pathname(filename, to_append)
    assert new_filename == Path("data/file_std.txt")
    assert new_filename.suffix == ".txt"
    assert new_filename.parent == Path("data")


def test_incremental_filename(tmp_path):
    base_filename = tmp_path / "output.txt"
    # Create the first file
    base_filename.touch()
    # Get an incremental filename
    new_filename = incremental_filename(base_filename)
    assert new_filename == tmp_path / "output-1.txt"
    # Create the second file
    new_filename.touch()
    # Get another incremental filename
    next_filename = incremental_filename(base_filename)
    assert next_filename == tmp_path / "output-2.txt"


def test_ensure_no_increment_if_file_not_exists(tmp_path):
    base_filename = tmp_path / "new_output.txt"
    # Ensure the file does not exist
    if base_filename.is_file():
        base_filename.unlink()
    # Get an incremental filename
    new_filename = incremental_filename(base_filename)
    assert new_filename == base_filename
    assert not new_filename.is_file()


def test_ensure_directory(tmp_path):
    dir_path = tmp_path / "new_directory"
    # Ensure the directory does not exist
    if dir_path.is_dir():
        dir_path.rmdir()
    # Use the function to ensure the directory exists
    ensure_directory(dir_path)
    assert dir_path.is_dir()
    # Call again to ensure it does not raise an error if the directory already exists
    ensure_directory(dir_path)


def test_overwrite_file_not_exists(tmp_path):
    # Test with non-existent file
    test_file = tmp_path / "non_existent.txt"

    result = overwrite(test_file)
    assert result is True
