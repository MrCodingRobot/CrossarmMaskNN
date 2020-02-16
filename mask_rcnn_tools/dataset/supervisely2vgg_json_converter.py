
"""
Author: Eduardo Davalos
Last Edit: 11/14/2019

Purpose of this file is to convert the annotations from Supervisely
annotations to VGG annotations.

Useful links:

[1] https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/

[2] https://docs.supervise.ly/import/local_files/supervisely/

[3] https://stackoverflow.com/questions/6591931/getting-file-size-in-python

    Size in VGG tool is from the actual file size

"""
#------------------------------------------------------------------------
# Imports

import json
import os
import pathlib
import tqdm

#------------------------------------------------------------------------

def rename_json_files():

    for json_file in os.listdir(supervisely_jsons):

        """
        Renaming JSON files to not have the .jpg or .JPG extension
        """

        new_json_file = json_file.replace(".json", ".jpg.json")

        if json_file.find(".jpg") == -1:

            json_absolute_path = os.path.join(supervisely_jsons, json_file)
            new_json_absolute_path = os.path.join(supervisely_jsons, new_json_file)

            os.rename(json_absolute_path, new_json_absolute_path)

    return None

#------------------------------------------------------------------------
# Main Code

supervisely_jsons = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\1DataSet4\ds\ann"
source_images_dir = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\1DataSet4\ds\img"

new_json_data = {}


for json_file in tqdm.tqdm(os.listdir(supervisely_jsons)):

    """
    Now with the clean JSON files, convert the data from Supervisely format to VGG annotator
    """

    # Getting the absolute path to the json file
    json_absolute_path = os.path.join(supervisely_jsons, json_file)
    json_file_object = pathlib.Path(json_file)

    # Getting the json data from the supervisely json files
    with open(json_absolute_path) as json_file:    
        json_data = json.load(json_file)

    # Getting jsons' corresponding source image
    image_filename = json_file_object.stem.replace(".jpg", "").replace(".JPG", "") + ".JPG"
    image_absolute_path = os.path.join(source_images_dir, image_filename)
    
    # Getting the source image's file size
    image_file_size = os.path.getsize(image_absolute_path)
    indexer = image_filename + str(image_file_size)

    # Generating new json file
    new_json_data[indexer] = {'filename': image_filename, 'size': image_file_size, 'regions': [], 'file_attributes': {}}

    # Getting the coordinate data from the json file
    if 'objects' in json_data.keys():
    
        for obj in json_data['objects']:

            region_data = {'shape_attributes': {"name": "polygon", "all_points_x": [], "all_points_y": []}, "region_attributes": {}}

            for points in obj['points']['exterior']:

                region_data['shape_attributes']['all_points_x'].append(points[0])
                region_data['shape_attributes']['all_points_y'].append(points[1])

            new_json_data[indexer]['regions'].append(region_data)

    with open("1DataSet4.json", "w") as outfile:
        json.dump(new_json_data, outfile)
