import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=False, default=True)
args = parser.parse_args()

path_object = Path(args.folder)

directory_list = list(path_object.iterdir())

print("Number of files: {}".format(len(directory_list)))