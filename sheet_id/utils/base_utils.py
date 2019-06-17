# pylint: disable=C0103
"""
Base utilities for getting lists of files, preprocessing annotations, and plotting visualizations
"""
import os
import glob
import pkg_resources
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def loadSettings():
    """
    Load settings file
    """
    settingsFilepath = pkg_resources.resource_filename('sheet_id', 'settings.yaml')
    with open(settingsFilepath, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as _:
            raise

def loadScoresDataset(path):
    """
    Returns a list of paths to scores (file extension: png)
    """
    return sorted(glob.glob(os.path.join(path, '*.png')))

def loadCellphoneScoresDataset(path):
    """
    Return a list of paths to cellphone scores (file extension: jpg)
    """
    return sorted(glob.glob(os.path.join(path, '*.jpg')))

def generateScoresDB(scorePaths):
    """
    Return a dictionary object
        {
            <filename without extension>: <representation> (e.g. np.array)
        }
    """
    db = {}
    for _, score in enumerate(scorePaths):
        img = cv2.imread(score, 0) # pylint: disable=E1101
        fileNameNoExt = os.path.splitext(os.path.split(score)[1])[0]
        db[fileNameNoExt] = img
    return db

def calculateMRR(ranks):
    """
    Return an MRR score based on the list of rank predictions
    """
    MRR = 0
    for rank in ranks:
        MRR += 1.0 / rank
    return MRR / len(ranks)

# pylint: disable=R0914
def generateSheetMaskAnnotation(img_path=loadSettings()['SCANNED_ANNOTATION_IMAGE_PATH'],
                                csv_path=loadSettings()['SCANNED_ANNOTATION_PATH'],
                                staff_height=92,
                                trimmed=True, plot=False):
    """
    Generate mask annotations from the bounding box annotations.

    Output:
        output - dictionary of images
    {
        filename: (image, mask, list of bounding boxes)
    }
    """

    # Read image paths
    if trimmed:
        img_paths = sorted(glob.glob(os.path.join(img_path, '*trimmed.png')))
    else:
        img_paths = sorted(glob.glob(os.path.join(img_path, '*[!trimmed].png')))

    # Read annotations
    df = pd.read_csv(csv_path, index_col=False)

    # Drop annotations not in the image annotation
    if trimmed:
        for i, row in df.iterrows():
            df.at[i, 'filename'] = df.at[i, 'filename'] + '_trimmed'
            filePath = os.path.join(img_path, df.at[i, 'filename'] + '.png')
            img_shape = cv2.imread(filePath, 0).shape # pylint: disable=E1101
            if df.at[i, 'vpix'] >= img_shape[0]:
                df.drop(i, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Generate output pairs
    output = {}
    for path in img_paths:
        filename = os.path.splitext(os.path.split(path)[1])[0]
        df_score = df[df['filename'] == filename]

        img = cv2.imread(path, 0) # pylint: disable=E1101

        staff_height_db = df_score['staff_height'].iloc[0]
        scaling_factor = staff_height / staff_height_db
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor) # pylint: disable=E1101

        mask = np.zeros(img.shape)

        boxes = []
        for i, row in df_score.iterrows():
            note_height = (row['staff_height'] / 8)
            (start_row, end_row) = (row['vpix'] - 1.2*note_height, row['vpix'] + 1.2*note_height)
            (start_col, end_col) = (row['hpix'] - 1.3*note_height, row['hpix'] + 1.3*note_height)
            start_row = int(start_row * scaling_factor)
            start_col = int(start_col * scaling_factor)
            end_row = int(end_row * scaling_factor)
            end_col = int(end_col * scaling_factor)

            boxes.append([start_col, start_row, end_col, end_row])
            mask[start_row:end_row,
                 start_col:end_col] = np.where(img[start_row:end_row,
                                                   start_col:end_col] is not None,
                                               29, 0)
            if plot:
                plt.figure(figsize=(20, 20))
                plt.subplot(1, 2, 1)
                plt.imshow(img, cmap='gray')
                plt.subplot(1, 2, 2)
                plt.imshow(mask, cmap='gray')
                plt.show()
        output[filename] = (img, mask, boxes)
    return output

def visualizeBoundingBoxes(img, boxes, figsize=(20, 20)):
    """
    Visualize bounding boxes on the provided image
    """
    _, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img, cmap='gray')
    patches = []
    for box in boxes:
        patches.append(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1]))
    pc = PatchCollection(patches, facecolor='None', edgecolor='r')
    ax.add_collection(pc)
    plt.show()
