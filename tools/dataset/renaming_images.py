import os
import argparse


def rename_directory(src, name):
    i = 0

    for filename in os.listdir(src):

        if filename.endswith(".JPG") is True:
            file_src = filename

            file_dst = name + str(i) + ".JPG"

            print("\nNumber: {}".format(i))
            print("File Source: {}\nFile Destination: {}".format(file_src, file_dst))

            os.rename(os.path.join(src, file_src), os.path.join(src, file_dst))
            i+=1

p = argparse.ArgumentParser()
p.add_argument("-s", "--source", required=True)
p.add_argument("-n", "--name", default="image")
args = p.parse_args()

rename_directory(args.source, args.name)
        