import os
import re
import dlib
import numpy as np
from pathlib import Path

# Defining the required portion of face landmarks
FACE = [i for i in range(3, 14)]

# Defining the required portion of nose landmarks
NOSE = [i for i in range(31, 36)]

# Defining mouth landmarks
MOUTH = [i for i in range(48, 68)]

# regex that identify the part section of the xml
REG_PART = re.compile("part name='[0-9]+'")

# regex that identify all the numbers (name, x, y) inside the part section
REG_NUM = re.compile("[0-9]+")


def main():
    # Original ibug model
    ibug_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), Path("labels_ibug_300W_train.xml"))

    # Test model
    ibug_test_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), Path("labels_ibug_300W_test.xml"))

    # Output model path
    hmd_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)), Path("hmd_face_landmarks.xml"))
    hmd_dat = os.path.join(os.path.dirname(os.path.abspath(__file__)), Path("hmd_face_landmarks.dat"))

    # Join all the points to be extracted in to a single array
    POINTS = np.concatenate((FACE, NOSE, MOUTH), axis=None)

    # Create the training xml for the new model with only the desired points
    slice_xml(input=ibug_xml, output=hmd_xml, points=POINTS)

    # Train the model
    train_model(input_xml=hmd_xml, output_model=hmd_dat)

    # Measure the model error on the testing annotations
    measure_model_error(test_xml=ibug_test_xml, model=hmd_dat)


def train_model(input_xml, output_model):
    """Trains the model

    :param input_xml: Path to the original xml file.
    :param output_model: Path to output the generated model
    """
    # Get the training options
    options = dlib.shape_predictor_training_options()
    options.tree_depth = 4
    options.nu = 0.1
    options.cascade_depth = 15
    options.feature_pool_size = 400
    options.num_test_splits = 50
    options.oversampling_amount = 5
    #
    options.be_verbose = True  # tells what is happening during the training
    options.num_threads = 4  # number of the threads used to train the model

    # finally, train the model
    dlib.train_shape_predictor(input_xml, output_model, options)


def slice_xml(input, output, points):
    """creates a new xml file based on the desired landmark-points.

    :param input: The input xml [input_xml_path] must be structured like the ibug annotation xml.
    :param output: The path to save the output xml file.
    :param points: Special set of landmarks to separate out.

    """
    input_file = open(input, "r")
    output_file = open(output, "w")
    points_to_extract = set(points)

    for line in input_file.readlines():
        finds = re.findall(REG_PART, line)

        # find the part section
        if len(finds) <= 0:
            output_file.write(line)
        else:
            # we are inside the part section
            # so we can find the part name and the landmark x, y coordinates
            name, x, y = re.findall(REG_NUM, line)

            # if is one of the point i'm looking for, write in the output file
            if int(name) in points_to_extract:
                output_file.write(f"      <part name='{int(name)}' x='{x}' y='{y}'/>\n")

    output_file.close()


def measure_model_error(test_xml, model):
    """Measures the error of the generated model

    :param test_xml: Path to the test xml file.
    :param model: Path to the generated model
    """
    error = dlib.test_shape_predictor(test_xml, model)
    print("Error of the model: {} is {}".format(model, error))


if __name__ == '__main__':
    main()
