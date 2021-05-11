from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import random
import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models as models
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.metrics import accuracy_score
from scipy.stats import sem
from sklearn.metrics import confusion_matrix

from collections import defaultdict

import albumentations as A
import warnings
warnings.filterwarnings("ignore")

def load_image(path):
    faces = crop_faces(path)
    if len(faces)>0:
        predict_gender = predict_gender_by_image(faces)
        predict_age = predict_age_by_image(faces)
        new_path = sub_faces(path, predict_gender, predict_age)
        return {
            'img_name': new_path,
            'predict_age': predict_age,
            'predict_gender': predict_gender
        }
    else:
        return {"error": "Лица не обнаружены"}

genderModel = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes= 2)

genderModel.load_state_dict(torch.load('./models/gender_model0.pth'))
genderModel.eval()

ageModel = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes= 4)

ageModel.load_state_dict(torch.load('./models/age_model0.pth'))
ageModel.eval()

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

class GenderDataset(Dataset):
    def __init__(self, path, image_files, labels_age, labels_gender, p_augment=0.5,  validation=False):
        self.path = path
        self.X = image_files
        self.y_age = labels_age
        self.y_gender = labels_gender
        self.resize = A.Resize(160, 160, always_apply=True)
        self.transform = trans
        
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = np.asarray(self.X[i])
        image = self.resize(image=image)['image']
        image = self.transform(image)
        label_age = self.y_age[i]
        label_gender = self.y_gender[i]
        
        return torch.tensor(image, dtype=torch.float), torch.tensor(label_gender, dtype=torch.long)

class AgeDataset(Dataset):
    def __init__(self, path, image_files, labels_age, labels_gender, p_augment=0.5,  validation=False):
        self.path = path
        self.X = image_files
        self.y_age = labels_age
        self.y_gender = labels_gender
        self.resize = A.Resize(160, 160, always_apply=True)
        self.transform = trans
        
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = np.asarray(self.X[i])
        image = self.resize(image=image)['image']
        image = self.transform(image)
        label_age = self.y_age[i]
        label_gender = self.y_gender[i]
        
        return torch.tensor(image, dtype=torch.float), torch.tensor(label_age, dtype=torch.long)

def predict_gender_by_image(image_path):
    device = torch.device("cpu")
    results = torch.empty(0).to(device)

    will_predict = [0]*len(image_path)
    dataset = GenderDataset('',
                            image_path, 
                            will_predict,
                            will_predict, 
                            validation=True)

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    for batch in loader:
        batch_data, batch_label = batch
        predict = genderModel(batch_data)

        results = torch.cat((results,
                             nn.functional.softmax(predict.detach(),dim=1)), 0)

        results = results.cpu().numpy()
    return results

def predict_age_by_image(image_path):
    device = torch.device("cpu")
    results = torch.empty(0).to(device)

    will_predict = [0]*len(image_path)
    dataset = AgeDataset('',
                            image_path, 
                            will_predict,
                            will_predict, 
                            validation=True)

    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    for batch in loader:
        batch_data, batch_label = batch
        predict = ageModel(batch_data)

        results = torch.cat((results,
                             nn.functional.softmax(predict.detach(),dim=1)), 0)

        results = results.cpu().numpy()
    return results

import numpy as np
import cv2

_age_labels = [
    'Lichinus 0-6',
    'Schegol 8-23',
    'Bumer 25-32',
    'Starper 35+'
]

def crop_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = Image.open(image_path)
    image = image.convert('RGB')

    basewidth = 1280
    wpercent = basewidth / image.size[0]
    hsize = int(image.size[1]*wpercent)
    image = image.resize((basewidth,hsize), Image.ANTIALIAS)

    image = np.array(image, 'uint8')

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=2, minSize=(160, 160))
    images = []
    for (x,y,w,h) in faces:
        _image = image[y:y+h,x:x+w]
        images.append(_image)

    return images

def sub_faces(image_path, predict_gender, predict_age):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image = Image.open(image_path)
    image = image.convert('RGB')

    basewidth = 1280
    wpercent = basewidth / image.size[0]
    hsize = int(image.size[1]*wpercent)
    image = image.resize((basewidth,hsize), Image.ANTIALIAS)

    image = np.array(image, 'uint8')

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=2, minSize=(160, 160))
    images = []
    
    i = 0
    for (x,y,w,h) in faces:
        
        gender_label = ""
        if predict_gender[i][0]>predict_gender[i][1]:
            gender_label = "Woman"
        else:
            gender_label = "Man"
        
        age_maximum = max(predict_age[i])
        age_max_item = 0
        for k, item in enumerate(predict_age[i]):
            #print(k, item)
            if item == age_maximum:
                age_max_item = k
                
        age_label = f'[{_age_labels[age_max_item]}]'
        
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),4)
        
        cv2.putText(image, age_label, 
            (x,y-15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (255,255,0),
            5)
        cv2.putText(image, gender_label, 
            (x,y+h+35), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            2,
            (255,255,0),
            5)
        i += 1

    image = Image.fromarray(image)

    img_name = image_path.split("/")
    if len(img_name) > 0:
        img_name = img_name[len(img_name)-1]
    else:
        img_name = img_name[0]

    image.save(f'server/static/'+img_name)

    return img_name
