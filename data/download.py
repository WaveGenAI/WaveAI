""" 
This script downloads the dataset
"""

import hashlib
import os
import shutil
import tempfile

import requests
from tqdm import tqdm
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--download_dir", type=str, required=True)
argparser.add_argument(
    "--nb_files_mgt",
    type=int,
    default=1000,
    required=False,
    help="Number of files to download from MTG",
)
argparser.add_argument(
    "--mtg",
    action="store_true",
    help="Download the dataset from the MTG server",
)

args = argparser.parse_args()


# Constants
CHUNK_SIZE = 512 * 1024
BASE_MUSIC_URL_JAMENTO = "https://cdn.freesound.org/mtg-jamendo/"


def download_file(url: str, download_path: str):
    """Download a file from the given url.

    Args:
        url (str): the url of the tar file
        download_path (str): the path to save the file
    """

    res = requests.get(url, stream=True, timeout=10)
    total = res.headers.get("Content-Length")
    if total is not None:
        total = int(total)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file_d:
        with tqdm(total=total, unit="B", unit_scale=True) as progressbar:
            for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
                tmp_file_d.write(chunk)
                progressbar.update(len(chunk))
        shutil.move(tmp_file_d.name, download_path)


def download_mtg(download_dir: str, nb_files: int = 1000):
    """Download the dataset from the MTG server.

    Args:
        download_dir (str): the directory to save the files
        nb_files (int, optional): the number of tar files to download. Defaults to 1000.

    Raises:
        ValueError: if the hash of one of the files is incorrect
    """

    file_sha256_tars = (
        BASE_MUSIC_URL_JAMENTO + "autotagging_moodtheme/audio/checksums_sha256.txt"
    )

    # download the checksums file
    res = requests.get(file_sha256_tars, timeout=10)
    res.raise_for_status()

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    checksums = res.text.split("\n")

    for i, checksum in enumerate(checksums):
        try:
            hash_tar_file, file = checksum.split(" ")
        except ValueError:
            continue

        print(f"Downloading {file}")

        url = BASE_MUSIC_URL_JAMENTO + f"autotagging_moodtheme/audio/{file}"
        download_file(url, os.path.join(download_dir, file))

        # check if the hash is correct
        with open(os.path.join(download_dir, file), "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash != hash_tar_file:
                raise ValueError(f"Hash mismatch for {file}")

        if (i + 1) >= nb_files:
            break

    for file in os.listdir(download_dir):
        if file.endswith(".tar") or file.endswith(".zip"):
            print(f"Extracting {file}")
            path = os.path.join(download_dir, file)
            shutil.unpack_archive(path, download_dir)
            os.remove(path)

    # move all the file in directory to the download_dir
    for directory in os.listdir(download_dir):
        dir_path = os.path.join(download_dir, directory)

        if not os.path.isdir(dir_path):
            continue

        for file in os.listdir(dir_path):
            try:
                shutil.move(os.path.join(dir_path, file), download_dir)
            except shutil.Error:
                shutil.move(
                    os.path.join(dir_path, file),
                    os.path.join(download_dir, f"{directory}_{file}"),
                )

        os.rmdir(dir_path)


if __name__ == "__main__":
    if args.mtg:
        download_mtg(download_dir=args.download_dir, nb_files=args.nb_files_mgt)
