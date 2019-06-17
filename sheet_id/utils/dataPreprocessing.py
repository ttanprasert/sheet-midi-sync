import glob
import os
from random import shuffle

def splitTrainValidation(deepscores_path, sub_folder='images_png', max_size=1000, test_size=30, npy_only=True):
    """
    Return two non-overlapped lists of images randomly shuffled.
    """
    images_list = []
    image_paths = None
    if npy_only:
        image_paths = os.path.join(deepscores_path, sub_folder, '*.npy')
    else:
        image_paths = os.path.join(deepscores_path, sub_folder, '*.png')
    images_list.extend(glob.glob(image_paths))
    
    for idx in range(len(images_list)):
        images_list[idx] = images_list[idx].replace('/' + sub_folder + '/', '/images_png/')

    assert test_size < max_size, "#{Test images} have to be smaller than #{All images}"
    assert max_size <= len(images_list), "Number of images isn't enough"

    test_image_list = images_list[0:test_size]
    train_image_list = images_list[test_size:max_size]

    return train_image_list, test_image_list
    
