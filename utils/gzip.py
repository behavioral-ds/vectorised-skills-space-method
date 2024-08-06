import os
import tarfile
import shutil


def gzip_directory(source_dir: str, output_filename: str):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


def uncompress_gzip(file_path):
    # Ensure the file has a .tar.gz extension
    if not file_path.endswith(".tar.gz"):
        raise ValueError("The file must have a .tar.gz extension")

    # Get the directory where the file is located
    dir_name = os.path.dirname(file_path)

    # Open the tarfile and extract its contents
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=dir_name)
