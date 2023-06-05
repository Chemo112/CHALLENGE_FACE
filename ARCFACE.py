from arcface.lib.models import ArcFaceModel
import arcface
from arcface.lib.utils import l2_norm
from astropy.utils.data import download_file
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
import errno

"""**QUESTE SONO LE CARTELLE DOVE YOLO O MTCNN AVRANNO SALVATO LE
FACCE**
"""

CROPPED_GPATH = '/OUTPUTS/YOLO/CROPPED_GALLERY/'
CROPPED_QPATH = '/OUTPUTS/YOLO/CROPPED_GALLERY/'

class ArcFace():
    def __init__(self, model_path = None):
        if model_path == None:
            from astropy.utils.data import download_file
            tflite_path = download_file("https://www.digidow.eu/f/datasets/arcface-tensorflowlite/model.tflite", cache=True)
        else:
            tflite_path = model_path

        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        print(self.input_details)
        self.output_details = self.interpreter.get_output_details()

    def calc_emb_single(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))
        img = img.astype(np.float32) / 255.
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        emb = l2_norm(output_data)
        return emb[0]

    def calc_emb_list(self, imgs):
        embs = []
        for img in imgs:
            embs.append(self.calc_emb_single(img))
        return embs

embedder=ArcFace()

len(os.listdir(CROPPED_QPATH))

"""**CREA GLI EMBEDDINGS PER LE IMMAGINI NELLA GALLERY**"""

gallery_encodings = {}
gallery_imgs_paths = {}
imgs = [CROPPED_GPATH + el for el in os.listdir(CROPPED_GPATH)]
i=0

for img_path in os.listdir(CROPPED_GPATH):
    print(img_path , i)
    img = CROPPED_GPATH + img_path
    #image = tf.cast(plt.imread(tmp), dtype=tf.float32)
    try:
      nome = img_path.split("/")[-1]
      est = nome.split(".")[1]
      nome = nome.split(".")[0]
      #perchè ho altri file nella cartella
      if est == "png" or est == "jpg":
        image_en = embedder.calc_emb_single(img)
        gallery_encodings[nome] = image_en
        gallery_imgs_paths[nome] = img
        i+=1
    except Exception as e:
      print(e, "Errore nell'estensione del file")
      pass

"""**CREA GLI EMBEDDINGS PER LE IMMAGINI NELLA CARTELLA QUERY**"""

query_encodings = {}
query_imgs_paths = {}
i=0
for img_path in os.listdir(CROPPED_QPATH):
    print(img_path , i)
    img = CROPPED_QPATH + img_path
    nome = img_path.split("/")[-1]
    est = nome.split(".")[1]
    nome = nome.split(".")[0]
    if est != "png" or est != "jpg":
      image_en = embedder.calc_emb_single(img)
      query_encodings[nome] = image_en
      query_imgs_paths[nome] = img
      i+=1

"""**CONFRONTO I VARI EMBEDDINGS E PRINTO I RISULTATI**"""

def get_distance_embeddings(emb1, emb2):
    diff = np.subtract(emb1, emb2)
    dist = np.sum(np.square(diff))
    return dist

risultati=dict()
for image_query in query_encodings:
  print("------")
  print(f"Query:{image_query}")
  fs_query = query_encodings[image_query]
  dists = []
  for image_gallery in gallery_encodings:
    fs_gallery = gallery_encodings[image_gallery]
    dist = np.linalg.norm(fs_query-fs_gallery)
    score = (dist, image_gallery)
    dists.append(score)
  dists.sort(key=lambda x: x[0])

  res = []
  for i in range(10):
    res.append(dists)
    print(f"{i}:{dists[i][1]}")

  risultati[image_query] = dists[0:10]

print(len(risultati))
print(risultati)

"""**VISUALIZZAZIONE RISULTATI**"""

import matplotlib.pyplot as plt
for query in risultati.keys():
  image = query_imgs_paths[query]
  plt.figure(figsize=(2,2))
  plt.subplot(1,1,1)
  plt.imshow(plt.imread(image))#se le immagini sono in jpg si VISUALIZZANO con img.astype("uint8")
  plt.title(query)
  plt.show()
  plt.figure(figsize=(20,8))
  i=0
  for gal in risultati[query]:
      plt.subplot(4, 10, i+1)
      plt.imshow(plt.imread(gallery_imgs_paths[gal[1]]))#se le immagini sono in jpg si VISUALIZZANO con img.astype("uint8")
      plt.axis('off')
      plt.title(str(i))
      i+=1
      if i == 10:
        break
  plt.show()

"""**CREA IL DIZIONARIO PER LA SUBMISSION**"""

submitdict = dict()

for query in risultati:
  vals = risultati[query]
  tmp = []
  for i in range(len(vals)):
    tmp.append(str(vals[i][1].split('_')[0])+'.jpg')   
  
  
  new_res_list = tmp
  submitdict[str(query.split('_')[0]) + ".jpg"] = new_res_list
  
print(len(submitdict))

mydata = dict()
mydata['groupname'] = "The Algorithm Avengers"
mydata['images'] = submitdict
print(mydata)

import json
import requests

def submit(results, url="https://competition-production.up.railway.app/results/"):
  res = json.dumps(results)
  response = requests.post(url, res)
  try:
    result = json.loads(response.text)
    print(f"accuracy is {result['results']}")
    return result
  except json.JSONDecodeError:
    print(f"ERROR: {response.text}")
    return None
    
submit(mydata)