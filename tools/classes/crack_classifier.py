# Library Imports
import pathlib

# Third-Party Imports
import cv2
import imutils
import numpy as np
import keras

#------------------------------------------------------------------
# Class

class CrackClassifier():

    def __init__(self, dict_para):

        # Checking if model path is valid
        assert pathlib.Path(dict_para["model_path"]).is_file() is True, "CrackClassifier Model Path Invalid"

        # If valid, store and continue
        self.dict_para = dict_para # Dictionary Parameters

        # Setup of Model
        self.model = keras.models.load_model(self.dict_para["model_path"])
        self.model.compile(loss="binary_crossentropy",
                           optimizer="adam",
                           metrics=["accuracy"])

        return None

    def predict_image_list(self, input_images_list):

        # Feeding input and visualizing output
        classifications = []

        for counter, test_image in enumerate(input_images_list):

            # Resizing image to (128,128) to improve performance
            test_image_saved = test_image
            test_image = cv2.resize(test_image,(128,128))
            test_image = np.reshape(test_image,[1,128,128,3])

            # Output classification
            classification = self.model.predict_classes(test_image)[0][0]

            print(classification)

            if classification == 0:
                tag = "Cracked"
            else:
                tag = "No cracked"

            classifications.append(tag)

            # Visualizing Output with OpenCV
            cv2.imshow("{} - {}".format(tag, counter), test_image_saved)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return classifications