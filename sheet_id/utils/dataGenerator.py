import numpy as np
import keras
import scipy.misc as misc
from random import shuffle, randint
import os
import xml.etree.ElementTree
from sheet_id.utils.dwd_utils import generateGroundTruthMaps

class DataGenerator(keras.utils.Sequence):
    """
    DataGenerator for keras training
    """
    def __init__(self, list_IDs, labels=None, batch_size=20, dim=(500,500), n_channels=1,
                 n_classes=124, shuffle=True, crop=True, crop_size=(500,500), load_npy=True):
        """
        Initialization
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.crop = crop
        self.crop_size = crop_size
        self.load_npy = load_npy
        self.on_epoch_end()

    def __len__(self):
        """
        Return the number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index+1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, energy_map, class_map, bbox_map, note_map, boxes = self.__data_generation(list_IDs_temp)

        return X, {
            'energy_map': np.expand_dims(energy_map, axis=-1),
            'class_map': np.expand_dims(class_map, axis=-1),
            'bbox_map': bbox_map,
            'note_map': note_map,
            'boxes': boxes
        }

    def on_epoch_end(self):
        """
        Update indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _load_xml_annotation(self, xml_path, limits=(None,None,None,None)):
        """
        Load XML file 

        xml_path - path to XML file
        limits = 4-tuple (x1,y1,x2,y2) specifying the regions to include bounding boxes
        """
        # Parse XML file
        tree = xml.etree.ElementTree.parse(xml_path)

        # Get all objects
        objs = tree.findall('object')

        # Create placeholders for bounding boxes
        num_objs = len(objs)
        boxes = []
 
        # Get the size of the entire sheet
        size = tree.find('size')
        height, width = float(size.find('height').text), float(size.find('width').text)

        # Get each bounding box object
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = np.floor(float(bbox.find('xmin').text)*width).astype(np.int16)
            y1 = np.floor(float(bbox.find('ymin').text)*height).astype(np.int16)
            x2 = np.floor(float(bbox.find('xmax').text)*width).astype(np.int16)
            y2 = np.floor(float(bbox.find('ymax').text)*height).astype(np.int16)

            # Elimiate boxes outside of the intended region
            if (limits[0] is not None) and (x2 < limits[0]): continue
            if (limits[2] is not None) and (x1 > limits[2]): continue
            if (limits[1] is not None) and (y2 < limits[1]): continue
            if (limits[3] is not None) and (y1 > limits[3]): continue

            new_x1 = x1 - limits[0]
            new_y1 = y1 - limits[1]
            new_x2 = x2 - limits[0] 
            new_y2 = y2 - limits[1]

            # Eliminate boxes whose center is outside
            center = ((new_x1+new_x2)/2, (new_y1+new_y2)/2)
            if center[0] < 0 or center[0] >= limits[2] - limits[0]: continue
            if center[1] < 0 or center[1] >= limits[3] - limits[1]: continue

            boxes.append([new_x1, new_y1, new_x2, new_y2, obj.find('name').text])
        
        return boxes

    def __data_generation(self, list_IDs_temp):
        """
        Generate data containing batch_size samples
        # X : (n_samples, *dim, n_channels)
        # y : (n_samples, *dim, n_channels)
        # boxes : list of boxes
        """
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=int)
        note_map = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=int)
        boxes = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            if self.load_npy:
                img = np.load(ID)
                annotation = np.load(ID.replace("/images_png/", "/pix_annotations_png/"))
                note = np.load(ID.replace("/images_png/", "/pix_annotations_png_new/"))
            else:
                img = misc.imread(ID, flatten=True)
                annotation = misc.imread(ID.replace("/images_png/", "/pix_annotations_png/"))
                note = misc.imread(ID.replace("/images_png/", "/pix_annotations_png_new/"))

            img = np.expand_dims(img, axis=-1)
            annotation = np.expand_dims(annotation, axis=-1)

            coord_0 = randint(0, (img.shape[0] - self.crop_size[0]))
            coord_1 = randint(0, (img.shape[1] - self.crop_size[1]))
            img = img[coord_0:(coord_0+self.crop_size[0]), coord_1:(coord_1+self.crop_size[1])]
            annotation = annotation[coord_0:(coord_0+self.crop_size[0]), coord_1:(coord_1+self.crop_size[1])]
            note_map[i,:] = np.expand_dims(note[coord_0:(coord_0+self.crop_size[0]), coord_1:(coord_1+self.crop_size[1])], axis=-1) 

            # load xml annotation
            xml_path = os.path.splitext(ID.replace('/images_png/', '/xml_annotations/'))[0] + '.xml'
            boxes_annotation = self._load_xml_annotation(xml_path, limits=(coord_1, coord_0, coord_1 + self.crop_size[1], coord_0 + self.crop_size[0]))

            X[i,:] = img
            y[i,:] = annotation
            boxes.append(boxes_annotation)

        energy_map_quantized, class_map, bbox_map = generateGroundTruthMaps(boxes, y, image_shape=img.shape)
        return X, energy_map_quantized, class_map, bbox_map, note_map, boxes
