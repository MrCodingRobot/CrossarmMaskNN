"""
Author: Eduardo Davalos
Last Edited: 11/14/2019

Supervisely Format: 

my_project
├── meta.json
├── dataset_name_01
│   ├── ann
│   │   ├── img_x.json
│   │   ├── img_y.json
│   │   └── img_z.json
│   └── img
│       ├── img_x.jpeg
│       ├── img_y.jpeg
│       └── img_z.jpeg
├── dataset_name_02
│   ├── ann
│   │   ├── img_x.json
│   │   ├── img_y.json
│   │   └── img_z.json
│   └── img
│       ├── img_x.jpeg
│       ├── img_y.jpeg
│       └── img_z.jpeg


NOTE:

[1] How to delete a directory:
https://stackoverflow.com/questions/6996603/delete-a-file-or-folder

    shutil.rmtree() deletes a directory and all its contents.

[2] How to create a directory:
https://stackabuse.com/creating-and-deleting-directories-with-python/

    os.makedirs(path)


Purpose of this file is to convert the annotations from VGG annotations
to Supervisely annotations.

"""

#----------------------------------------------------------------------

DIRECTORY_NAME = "home"

#----------------------------------------------------------------------
# Imports

import json
import shutil
import pathlib
import os
import cv2
import tqdm

#----------------------------------------------------------------------
# Utilities

#----------------------------------------------------------------------
# Main Code

for dir in [r'home\train', r'home\val']:

    #print(os.listdir(dir))

    vgg_json_file = os.path.join(dir, 'via_region_data.json')
    #print(vgg_json_file)

    with open(vgg_json_file) as json_file:
        datas = json.load(json_file)

    for data in tqdm.tqdm(datas):

        #print(data)

        filename_JPG = datas[data]['filename']
        filename_jpg = filename_JPG.replace(".JPG", ".jpg")

        #print(filename_JPG)

        image_source_location = os.path.join(dir, "img" + "\\" + filename_JPG)
        #print(image_source_location)

        image = cv2.imread(image_source_location)
        h, w, _ = image.shape

        # Writting output JSON file
        new_json_location = os.path.join(dir, "ann" + "\\" + filename_jpg + ".json")
        #print(new_json_location)

        new_json_data = {"description": "",
                         "tags": [],
                         "size": {
                             "height": h,
                             "width": w
                         },
                         "objects": []}

        for region in datas[data]["regions"]:

            x_points = region["shape_attributes"]["all_points_x"]
            y_points = region["shape_attributes"]["all_points_y"]

            #print("x points")
            #print(x_points)
            #print("y points")
            #print(y_points)

            region_data = {"description": "",
                "bitmap": None,
                "tags": [],
                "classTitle": "Cross arm",
                "points": {
                    "exterior": [],
                    "interior": []}
                }

            for i in range(len(x_points)):

                region_data["points"]["exterior"].append([x_points[i], y_points[i]])

            new_json_data["objects"].append(region_data)

        with open(new_json_location, "w") as outfile:
            json.dump(new_json_data, outfile)

    









