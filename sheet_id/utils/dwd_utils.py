# pylint: disable=C0103
"""
Utility functions for DeepWatershed Detector
"""
import math
import random
from itertools import product
import numpy as np
from PIL import Image
from sheet_id.utils.eval_utils import filterBoxes

# pylint: disable=R0914
def generateGroundTruthMaps(bboxes, annotations, image_shape=(250, 250), e_max=10, r=15):
    """
    Generate ground truth quantized energy map given the bounding boxes.
    This optimized version takes O(r^2 * |n_objects|) instead of O(row*col*|n_objects|).
    """
    energy_map = np.zeros((len(bboxes), image_shape[0], image_shape[1]))
    bbox_map = np.zeros((energy_map.shape[0], energy_map.shape[1], energy_map.shape[2], 2))

    # Create discrete energy bins
    bins = np.linspace(0, e_max + np.finfo(np.float32).eps, e_max+1)

    for idx, _ in enumerate(bboxes):
        for obj in bboxes[idx]:
            minrow, maxrow = max(0, obj[1] - r), min(image_shape[0], obj[3] + r)
            mincol, maxcol = max(0, obj[0] - r), min(image_shape[1], obj[2] + r)
            for row in range(minrow, maxrow):
                for col in range(mincol, maxcol):
                    center = (obj[0] + obj[2])/2, (obj[1] + obj[3])/2
                    energy_map[idx, row, col] = max(energy_map[idx, row, col],
                                                    e_max * (1 - np.sqrt((row - center[1])**2 \
                                                                         + (col - center[0])**2)
                                                             / r))
                    if energy_map[idx, row, col] >= bins[1]:
                        bbox_map[idx, row, col, 0] = obj[3] - obj[1] # height
                        bbox_map[idx, row, col, 1] = obj[2] - obj[0] # width

    # Discretize energy map
    energy_map_quantized = np.digitize(energy_map, bins) - 1
    # Generate class map
    class_map = (energy_map_quantized > 0) * (annotations[:, :, :, 0])
    return energy_map_quantized, class_map, bbox_map

def generateFullpagePrediction(model, img, window_size=(500, 500), step_size=(450, 450)):
    """
    Run the model using sliding window to obtain the DWD prediction on an image
    larger than (500,500)
    """
    (n_rows, n_cols) = img.shape
    n_steps_r = 1 + math.ceil((n_rows - window_size[0]) / step_size[0])
    n_steps_c = 1 + math.ceil((n_cols - window_size[1]) / step_size[1])

    energy_map_final = np.empty((window_size[0] + (n_steps_r - 1) * step_size[0],
                                 window_size[1] + (n_steps_c - 1) * step_size[1]))
    class_map_final = np.empty((window_size[0] + (n_steps_r - 1) * step_size[0],
                                window_size[1] + (n_steps_c - 1) * step_size[1]))
    bbox_map_final = np.empty((window_size[0] + (n_steps_r - 1) * step_size[0],
                               window_size[1] + (n_steps_c - 1) * step_size[1], 2))

    for i in range(n_steps_r):
        for j in range(n_steps_c):
            # Load patch
            start_row = i * step_size[0]
            start_col = j * step_size[1]
            end_row = min(n_rows, start_row + window_size[0])
            end_col = min(n_cols, start_col + window_size[1])

            img_patch = np.ones(window_size) * 255
            img_patch[:(end_row-start_row), :(end_col-start_col)] = img[start_row:end_row,
                                                                        start_col:end_col]

            # Model prediction + conversion
            energy_map, class_map, bbox_map = model.predict(img_patch.reshape(1, window_size[0],
                                                                              window_size[1], 1))
            # binarize image
            energy_map = np.argmax(energy_map, axis=-1)[0]
            class_prediction_img = np.argmax(class_map, axis=-1)[0]
            bbox_prediction = bbox_map[0, :, :, :]

            energy_map_final[start_row:start_row+window_size[0],
                             start_col:start_col+window_size[1]] = energy_map
            class_map_final[start_row:start_row+window_size[0],
                            start_col:start_col+window_size[1]] = class_prediction_img
            bbox_map_final[start_row:start_row+window_size[0],
                           start_col:start_col+window_size[1], :] = bbox_prediction

    energy_map_final = energy_map_final[:n_rows, :n_cols]
    class_map_final = class_map_final[:n_rows, :n_cols]
    bbox_map_final = bbox_map_final[:n_rows, :n_cols, :]

    return energy_map_final, class_map_final, bbox_map_final

# Array based union find data structure

# P: The array, which encodes the set membership of all the elements
# pylint: disable=C0111
class UFarray:
    def __init__(self):
        # Array which holds label -> set equivalences
        self.P = []

        # Name of the next label, when one is created
        self.label = 0

    def makeLabel(self):
        r = self.label
        self.label += 1
        self.P.append(r)
        return r

    # Makes all nodes "in the path of node i" point to root
    def setRoot(self, i, root):
        while self.P[i] < i:
            j = self.P[i]
            self.P[i] = root
            i = j
        self.P[i] = root

    # Finds the root node of the tree containing node i
    def findRoot(self, i):
        while self.P[i] < i:
            i = self.P[i]
        return i

    # Finds the root of the tree containing node i
    # Simultaneously compresses the tree
    def find(self, i):
        root = self.findRoot(i)
        self.setRoot(i, root)
        return root

    # Joins the two trees containing nodes i and j
    # Modified to be less agressive about compressing paths
    # because performance was suffering some from over-compression
    def union(self, i, j):
        if i != j:
            root = self.findRoot(i)
            rootj = self.findRoot(j)
            if root > rootj:
                root = rootj
            self.setRoot(j, root)
            self.setRoot(i, root)

    def flatten(self):
        for i in range(1, len(self.P)):
            self.P[i] = self.P[self.P[i]]

    def flattenL(self):
        k = 1
        for i in range(1, len(self.P)):
            if self.P[i] < i:
                self.P[i] = self.P[self.P[i]]
            else:
                self.P[i] = k
                k += 1

def find_connected_comp(input_map):
    data = input_map
    height, width = input_map.shape

    # Union find data structure
    uf = UFarray()

    #
    # First pass
    #

    # Dictionary of point:label pairs
    labels = {}

    for y, x in product(range(height), range(width)):

        #
        # Pixel names were chosen as shown:
        #
        #   -------------
        #   | a | b | c |
        #   -------------
        #   | d | e |   |
        #   -------------
        #   |   |   |   |
        #   -------------
        #
        # The current pixel is e
        # a, b, c, and d are its neighbors of interest
        #
        # 255 is white, 0 is black
        # White pixels part of the background, so they are ignored
        # If a pixel lies outside the bounds of the image, it default to white
        #

        # If the current pixel is white, it's obviously not a component...
        if data[y, x] == 255:
            pass

        # If pixel b is in the image and black:
        #    a, d, and c are its neighbors, so they are all part of the same component
        #    Therefore, there is no reason to check their labels
        #    so simply assign b's label to e
        elif y > 0 and data[y - 1, x] == 0:
            labels[y, x] = labels[(y - 1, x)]

        # If pixel c is in the image and black:
        #    b is its neighbor, but a and d are not
        #    Therefore, we must check a and d's labels
        elif x + 1 < width and y > 0 and data[y - 1, x + 1] == 0:

            c = labels[(y - 1, x + 1)]
            labels[y, x] = c

            # If pixel a is in the image and black:
            #    Then a and c are connected through e
            #    Therefore, we must union their sets
            if x > 0 and data[y - 1, x - 1] == 0:
                a = labels[(y - 1, x - 1)]
                uf.union(c, a)

            # If pixel d is in the image and black:
            #    Then d and c are connected through e
            #    Therefore we must union their sets
            elif x > 0 and data[y, x - 1] == 0:
                d = labels[(y, x - 1)]
                uf.union(c, d)

        # If pixel a is in the image and black:
        #    We already know b and c are white
        #    d is a's neighbor, so they already have the same label
        #    So simply assign a's label to e
        elif x > 0 and y > 0 and data[y - 1, x - 1] == 0:
            labels[y, x] = labels[(y - 1, x - 1)]

        # If pixel d is in the image and black
        #    We already know a, b, and c are white
        #    so simpy assign d's label to e
        elif x > 0 and data[y, x - 1] == 0:
            labels[y, x] = labels[(y, x - 1)]

        # All the neighboring pixels are white,
        # Therefore the current pixel is a new component
        else:
            labels[y, x] = uf.makeLabel()

    #
    # Second pass
    #

    uf.flatten()

    colors = {}

    # Image to display the components in a nice, colorful way
    output_img = Image.new("RGB", (height, width))
    outdata = output_img.load()

    for (y, x) in labels:

        # Name of the component the current point belongs to
        component = uf.find(labels[(y, x)])

        # Update the labels with correct information
        labels[(y, x)] = component

        # Associate a random color with this component
        if component not in colors:
            colors[component] = (random.randint(0, 255), random.randint(0, 255),
                                 random.randint(0, 255))

        # Colorize the image
        outdata[y, x] = colors[component]

    return (labels, output_img)

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if not boxes.size:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while idxs.size:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

# pylint: disable=R0913
def inferBoundingBoxes(energy_map, class_map, bbox_map, energy_nbins=10,
                       threshold_binarization=5, classToFilter=(29, ),
                       non_max_suppression_factor=0.25):
    """
    Generate a list of bounding boxes based on the outputs from the DWD model.

    Inputs:
    energy_map - a quantized energy map of shape (rows, cols)
    class_map - a quantized class prediction map of shape (rows, cols)
    bbox_map - a bounding box prediction map of shape (rows, cols, 2) [width and height]
    """

    # Binarize energy map
    energy_map_binarized = 255 * (energy_map < threshold_binarization)

    # Find components based on the energy map
    labels, _ = find_connected_comp(energy_map_binarized)

    counter = {}
    center = {}
    class_pred = {}
    bbox = {}
    for label in labels:
        component_id = labels[label]

        # Initialize component
        if component_id not in counter:
            counter[component_id] = 0
            center[component_id] = (0, 0)
            class_pred[component_id] = []
            bbox[component_id] = (0, 0)

        # Compute the center, class prediction, bounding prediction
        counter[component_id] += 1
        center[component_id] = (center[component_id][0] + label[0],
                                center[component_id][1] + label[1])
        class_pred[component_id].append(class_map[label[0], label[1]])
        bbox[component_id] = (bbox[component_id][0] + bbox_map[label[0], label[1], 0],
                              bbox[component_id][1] + bbox_map[label[0], label[1], 1])

    # Normalize each prediction by number of elements in each component
    for center_id in center:
        center[center_id] = (center[center_id][0] / counter[center_id],
                             center[center_id][1] / counter[center_id])
        class_pred[center_id] = max(set(class_pred[center_id]),
                                    key=class_pred[center_id].count)
        bbox[center_id] = (bbox[center_id][0] / counter[center_id],
                           bbox[center_id][1] / counter[center_id])

    boxes = []
    # filter prediction boxes
    for _, obj in enumerate(center.keys()):
        boxes.append([center[obj][1] - bbox[obj][1] / 2, center[obj][0] - bbox[obj][0] / 2,
                      center[obj][1] + bbox[obj][1] / 2, center[obj][0] + bbox[obj][0] / 2,
                      class_pred[obj]])

    # Filter boxes and perform non-max suppression
    boxes = filterBoxes(boxes, classToFilter)
    boxes = np.array(boxes)
    if non_max_suppression_factor is not None:
        suppressed_boxes = non_max_suppression_fast(boxes, non_max_suppression_factor)
    else:
        suppressed_boxes = boxes

    # Compute the score based on the energy at the center point
    scores = []
    for _, box in enumerate(suppressed_boxes):
        center_point_c = int((box[0] + box[2]) / 2)
        center_point_r = int((box[1] + box[3]) / 2)
        scores.append(energy_map[center_point_r, center_point_c] / energy_nbins)

    return suppressed_boxes, scores
