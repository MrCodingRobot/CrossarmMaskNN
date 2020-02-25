# This script randomly grabs a directory and randomly sortes images into the 
# train and val directories.

import os
import random
import shutil

#-----------------------------------------------------------------------------
# Constants

new_dir = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\crossarm\Group3"

train_dir = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\crossarm\train"
val_dir = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\crossarm\val"

#-----------------------------------------------------------------------------
# Classes

class SetContainer():

    def __init__(self, dataset_list):

        self.dataset_dic = {}
        self.dataset_list = dataset_list

        for dataset in dataset_list:
            self.dataset_dic[dataset.label] = dataset

        return None

class DataSet():

    def __init__(self, label, directory):

        self.label = label
        self.dir = directory

        return None

#------------------------------------------------------------------------------
# Functions

def generate_random_number_list(needed_images, num_of_jpgs):

    random_list = []

    while True:

        rand_num = random.randint(0,num_of_jpgs-1)

        if rand_num not in random_list:
            random_list.append(rand_num)

        if len(random_list) == needed_images:
            break

    return random_list

def move_to_train(dataset, random_list):

    global train_dir

    for file_num in random_list:

        print("****************")
        filename = dataset.jpg_files[file_num]
        print(filename)
        src = os.path.join(dataset.dir, filename)
        print(src)
        dst = os.path.join(train_dir, filename)
        print(dst)

        shutil.move(src, dst)

    return None

#------------------------------------------------------------------------------
# Main

dirs = SetContainer([DataSet('train', train_dir), DataSet('val', val_dir), DataSet('new', new_dir)])
dirs.total_num_of_jpgs = 0

for dataset in dirs.dataset_list:

    print("****************")
    print("Key: {} - Value: {}".format(dataset.label, dataset.dir))
    
    files = os.listdir(dataset.dir)
    dataset.jpg_files = [file for file in files if file.endswith(".JPG")]
    
    dataset.num_of_jpgs = len(dataset.jpg_files)
    dirs.total_num_of_jpgs += dataset.num_of_jpgs
    
    print("Number of JPEG: {}".format(dataset.num_of_jpgs))

print("\n#########################")
print("Total Number of JPEGS: {}".format(dirs.total_num_of_jpgs))

train_val_split = 0.7 # 70% train, 30% val
train_images = round(dirs.total_num_of_jpgs * train_val_split)

new_train_images = train_images - dirs.dataset_dic['train'].num_of_jpgs
print("new train images: {}".format(new_train_images))

#print("new val images: {}".format())

new_selected_images = generate_random_number_list(new_train_images, dirs.dataset_dic['new'].num_of_jpgs)

#print("Randomly selected files for exc: {}".format(exc_selected_images))
#print("Randomly selected files for pic: {}".format(pic_selected_images))

move_to_train(dirs.dataset_dic['new'], new_selected_images)
#move_to_train(dirs.dataset_dic['pic'], pic_selected_images)


# Moving images to train











