from tqdm import tqdm
from multiprocessing import Pool, freeze_support
import glob
import os 
import scipy.misc as misc
import numpy as np

img_list = sorted(glob.glob('/data/mirlab/DeepScores/pix_annotations_png/*.png'))[0:1000]

def progresser(n):         
    start = (n * len(img_list) // 10)
    end = ((n+1) * len(img_list) // 10)
    text = "progresser #{}".format(n)
    for i in tqdm(range(start,end), desc=text, position=n):
        img_path = img_list[i]
        filename = os.path.split(img_path)[1]
        filename_noext = os.path.splitext(filename)[0]
        img = misc.imread(img_path, flatten=True)
        np.save('/data/mirlab/DeepScores/pix_annotations_png/{:}.npy'.format(filename_noext), img)
        img = misc.imread('/data/mirlab/DeepScores/images_png/{:}.png'.format(filename_noext), flatten=True)
        np.save('/data/mirlab/DeepScores/images_png/{:}.npy'.format(filename_noext), img)

L = list(range(10))
Pool(len(L)).map(progresser, L)
