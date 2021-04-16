"""
Downloads the MovieLens dataset and saves it as an artifact
"""
import tempfile
import os
import zipfile
import mlflow
import click
from google_drive_downloader import GoogleDriveDownloader as gdd

def load_raw_data():
    with mlflow.start_run() as mlrun:
        local_dir = tempfile.mkdtemp()
        local_filename = os.path.join(local_dir, "artist_dataset.zip")
        print("Downloading input to %s" % ( local_filename))
        gdd.download_file_from_google_drive(file_id='1BRerZeIr25uE9PhwRPbRhgJ1-xm6IF_A',
                                    dest_path=local_filename,
                                    unzip=True)

        artist_lib = os.path.join(local_dir, "artist_dataset")

        print("Uploading artist_lib: %s" % artist_lib)
        mlflow.log_artifact(artist_lib, "artist_lib")


if __name__ == "__main__":
    load_raw_data()
