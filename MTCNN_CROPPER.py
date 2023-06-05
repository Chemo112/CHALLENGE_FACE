import facealignment
import tensorflow as tf
import os
from PIL import Image
import numpy as np
from mtcnn import MTCNN
import cv2
import facealignment
import time
#REMEMBER TO PIP INSTALL facealignment

GENERAL_GALLERY_PATH = "/INPUTS/GALLERY/"
GENERAL_QUERY_PATH = "/INPUTS/QUERY/"

MTCNN_SAVE_GPATH = "/OUTPUTS/GALLERY/"
MTCNN_SAVE_QPATH = "/OUTPUTS/QUERY/"

gallery_list = [GENERAL_GALLERY_PATH + path for path in os.listdir(GENERAL_GALLERY_PATH)]
query_list = [GENERAL_QUERY_PATH + path for path in os.listdir(GENERAL_QUERY_PATH)]

start = time.time()
print(start)

def crop(img_list, savepath):
    for el in img_list:
        multi_face = cv2.imread(el)
        tool = facealignment.FaceAlignmentTools()
        multi_face = cv2.cvtColor(multi_face, cv2.COLOR_BGR2RGB)
        aligned_imgs = tool.align(multi_face, (160, 160), allow_multiface=True, central_face=False)
        name = el.split("/")[-1]
        name = name.split(".")[0]
        i = 0
        try:
            for img in aligned_imgs:
                filename = savepath + name + "_" + str(i) + ".png"
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, img)
                i += 1
                print(i, name)
        except Exception as e:
            print("Nessuna faccia riconosciuta", name, e)
            pass


crop(gallery_list, MTCNN_SAVE_GPATH)
crop(query_list, MTCNN_SAVE_QPATH)

end = time.time()
print("EXECUTION TIME:", end - start)
