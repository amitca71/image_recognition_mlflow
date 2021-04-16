"""
Converts the raw CSV form to a Parquet form with just the columns we want
"""
import tempfile
import os
import mlflow
import click
import splitfolders

@click.command(
    help="Given a CSV file (see load_raw_data), transforms it into Parquet "
    "in an mlflow artifact called 'ratings-parquet-dir'"
)
@click.option("--aertist_dir")

def etl_data(aertist_dir):
    with mlflow.start_run() as mlrun:
        in_local_dir = tempfile.mkdtemp()
        out_local_dir=tempfile.mkdtemp()
        print(out_local_dir)
        print("aertist_dir=%s" % aertist_dir[1:-1])

        splitfolders.ratio(aertist_dir[1:-1], output=out_local_dir, seed=1337, ratio=(.8, .2), group_prefix=None)
    
        artist_train = os.path.join(out_local_dir, "train")
        artist_validation = os.path.join(out_local_dir, "val")
#        print("Uploading artist_split_lib: %s" % artist_train)
        mlflow.log_artifact(artist_train, "artist_train")  
        mlflow.log_artifact(artist_validation, "artist_val")
        
        

if __name__ == "__main__":
    etl_data()
