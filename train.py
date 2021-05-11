#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys

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

dataset_path = 'dataset/'
image_path = dataset_path+'faces/'
models_path = 'models/'
random_seed = 1120
trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
group_4class_age = {
    'Личинус (0-6)':    ['(0, 2)', '2', '(4, 6)', '3'],
    'Щегол (8-23)': ['(8, 12)', '(15, 20)', '(8, 23)', '23', '22', '13'],
    'Бумер (25-32)':  ['(25, 32)', '(27, 32)', '32', '34', '29'],
    'Старпёр 35+':  ['(48, 53)', '(60, 100)', '55','56','57', '58', '(38, 42)','(38, 43)', '(38, 48)', '35', '36', '46', '45', '42'],
}

def start():
    parametr = "--all"
    if len (sys.argv) > 1:
        parametr = sys.argv[1]

    seed_everything(random_seed)

    # готовим данные
    test_fold, all_age_group = read_test_data()
    df = create_test_folds(test_fold, all_age_group, map_age(group_4class_age))
    all_fold, train_fold = create_train_folds(test_fold)
    enumerate_features(test_fold, train_fold, all_fold, random_seed)
    print("Данные подготовлены")

    # тут гиперпараметры
    kfold = 2
    batchsize = 64
    lr_age = 3e-5
    lr_gender= 2e-5
    num_epochs = 5
    p_augment = 0.0

    device = torch.device("cpu")
    num_age_classes = train_fold[0].age.value_counts().shape[0]
    num_gender_classes = train_fold[0].gender.value_counts().shape[0]

    # обучение "пол"
    if parametr == "--all" or parametr == "--gender":
        train_gender(
            kfold,
            batchsize,
            lr_age,
            lr_gender,
            num_epochs,
            p_augment,
            device, num_age_classes, num_gender_classes,
            test_fold, train_fold, random_seed
        )

    # обучение "возраст"
    if parametr == "--all" or parametr == "--age":
        train_age(
            kfold,
            batchsize,
            lr_age,
            lr_gender,
            num_epochs,
            p_augment,
            device, num_age_classes, num_gender_classes,
            test_fold, train_fold, random_seed
        )

def enumerate_features(test_fold, train_fold, all_fold, random_seed):
    # приводим значения к числовым
    gender_to_label_map = {
        'f' : 0,
        'm' : 1
    }

    # приводим значения к числовым
    age_to_label_map = {
        'Личинус (0-6)' :0,
        'Щегол (8-23)' :1,
        'Бумер (25-32)' :2,
        'Старпёр 35+' :3
    }

    label_to_age_map = {value: key for key, value in age_to_label_map.items()}
    label_to_gender_map = {value: key for key, value in gender_to_label_map.items()}

    all_fold['age'].replace(age_to_label_map, inplace=True)
    all_fold['gender'].replace(gender_to_label_map, inplace=True)

    for i, fold in enumerate(train_fold):
        fold['age'].replace(age_to_label_map, inplace=True)
        fold['gender'].replace(gender_to_label_map, inplace=True)

    for i, fold in enumerate(test_fold):
        fold['age'].replace(age_to_label_map, inplace=True)
        fold['gender'].replace(gender_to_label_map, inplace=True)

    for fold in range(5):
        sss = StratifiedShuffleSplit(n_splits=10, random_state=random_seed)
        
        train_data = train_fold[fold]['image_path'].copy().reset_index(drop=True).to_list()
        train_gender_label = train_fold[fold]['gender'].copy().reset_index(drop=True).to_list()
        train_age_label = train_fold[fold]['age'].copy().reset_index(drop=True).to_list()
        train_idx, val_idx = list(sss.split(train_data, train_age_label))[0]
        
        print(f'Training data: {len(train_idx)}')
        print(f'Val. data: {len(val_idx)}')

#группирование, по 4ем категориям
def create_train_folds(test_fold):
    train_fold = [pd.concat([test_fold[1],test_fold[2],test_fold[3],test_fold[4]],ignore_index=True)]
    train_fold.append(pd.concat([test_fold[0],test_fold[2],test_fold[3],test_fold[4]],ignore_index=True))
    train_fold.append(pd.concat([test_fold[0],test_fold[1],test_fold[3],test_fold[4]],ignore_index=True))
    train_fold.append(pd.concat([test_fold[0],test_fold[1],test_fold[2],test_fold[4]],ignore_index=True))
    train_fold.append(pd.concat([test_fold[0],test_fold[1],test_fold[2],test_fold[3]],ignore_index=True))

    # датафрейм со всем фолдами
    all_fold = pd.concat([test_fold[0],test_fold[1],test_fold[2],test_fold[3],test_fold[4]],ignore_index=True)
    return all_fold, train_fold

def create_test_folds(test_fold, all_age_group, age_to_label):
    for fold, df in enumerate(test_fold):
            
        #полный путь до изображения
        df['image_path'] = image_path + df['user_id'] + '/coarse_tilt_aligned_face.' + df['face_id'].astype('str') + '.' + df['original_image']
    
        #удаляем лишние колонки
        df.drop(['user_id', 'original_image', 'face_id', 'x', 'y', 'dx', 
            'dy', 'tilt_ang', 'fiducial_yaw_angle', 'fiducial_score'], axis=1, inplace=True)
        
        df.gender = df.gender.astype(str)
        df.age = df.age.astype(str)
        
        #удаляем строки с пустыми значениями
        df.drop(df[df.gender == 'u'].index, inplace=True)
        df.drop(df[df.gender == 'nan'].index, inplace=True)
        df.drop(df[df.age == 'None'].index, inplace=True)
        
        #лейбл возраста -> лейбл возрастной категории
        include_age = list(age_to_label.keys())
        exclude_age = list(set(all_age_group) - set(include_age))
        
        #удаляем если вдруг не попало ни в одну возрастную категорию
        for exc_age in exclude_age:
            df.drop(df.loc[df['age']==exc_age].index, inplace=True)
            
        #упорядочиваем индексы по возрасту
        df['age'] = df['age'].apply(lambda x: age_to_label[x])
    return df

def read_test_data():
    test_fold = [pd.read_csv(f"{dataset_path}fold_0_data.txt",sep = "\t")]
    test_fold.append(pd.read_csv(f"{dataset_path}fold_1_data.txt",sep = "\t"))
    test_fold.append(pd.read_csv(f"{dataset_path}fold_2_data.txt",sep = "\t"))
    test_fold.append(pd.read_csv(f"{dataset_path}fold_3_data.txt",sep = "\t"))
    test_fold.append(pd.read_csv(f"{dataset_path}fold_4_data.txt",sep = "\t"))

    all_age_group = pd.concat([test_fold[0],test_fold[1],test_fold[2], test_fold[3],test_fold[4]]).age.value_counts()
    all_age_group = list(all_age_group.index)

    return test_fold, all_age_group

def map_age(group_to_age):
    age_to_group = {}
    for group in group_to_age.keys():
        age = group_to_age[group]
        for aa in age:
            age_to_group[aa] = group
    return age_to_group

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
        image = Image.open(self.path + self.X[i])
        image = np.asarray(image)
        image = self.resize(image=image)['image']
        image = self.transform(image)
        label_age = self.y_age[i]
        label_gender = self.y_gender[i]
        
        return torch.tensor(image, dtype=torch.float), torch.tensor(label_age, dtype=torch.long)

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
        image = Image.open(self.path + self.X[i])
        image = np.asarray(image)
        image = self.resize(image=image)['image']
        image = self.transform(image)
        label_age = self.y_age[i]
        label_gender = self.y_gender[i]
        
        return torch.tensor(image, dtype=torch.float), torch.tensor(label_gender, dtype=torch.long)

def train_gender(
            kfold,
            batchsize,
            lr_age,
            lr_gender,
            num_epochs,
            p_augment,
            device, num_age_classes, num_gender_classes,
            test_fold, train_fold, random_seed):
    all_accuracy_gender = []
    all_val_loss_gender = []
    all_stat_fold = []
    
    for fold in range(kfold):
        all_stat = defaultdict(list)
        # image paths
        train_data = train_fold[fold]['image_path'].copy().reset_index(drop=True).to_list()
        test_data  = test_fold[fold]['image_path'].copy().reset_index(drop=True).to_list()
    
        #get label
        train_age_label = train_fold[fold]['age'].copy().reset_index(drop=True).to_list()
        train_gender_label = train_fold[fold]['gender'].copy().reset_index(drop=True).to_list()
        test_age_label = test_fold[fold]['age'].copy().reset_index(drop=True).to_list()
        test_gender_label = test_fold[fold]['gender'].copy().reset_index(drop=True).to_list()
    
        #create train-validation stratified split
        sss = StratifiedShuffleSplit(n_splits=10, random_state=random_seed)
    
        #split based on age, more balanced for both age and gender
        train_idx, val_idx = list(sss.split(train_data, train_gender_label))[0]
    
        train_idx = list(train_idx)
        val_idx = list(val_idx)
    
        #create dataloader for gender
        train_dataset = GenderDataset('', 
                                          list(np.array(train_data)[train_idx]), 
                                          list(np.array(train_age_label)[train_idx]),
                                          list(np.array(train_gender_label)[train_idx]),
                                          p_augment = p_augment)
        val_dataset   = GenderDataset('', 
                                          list(np.array(train_data)[val_idx]), 
                                          list(np.array(train_age_label)[val_idx]),
                                          list(np.array(train_gender_label)[val_idx]),
                                          validation=True)
        test_dataset = GenderDataset('', 
                                  test_data, 
                                  test_age_label,
                                  test_gender_label, 
                                  validation=True)
    
    
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    
        val_gender_label = list(np.array(train_gender_label)[val_idx])
        val_age_label = list(np.array(train_age_label)[val_idx])
    
    
        model = InceptionResnetV1(
                        classify=True,
                        pretrained='vggface2',
                        num_classes=num_gender_classes)
        model = model.to(device)
    
        #optimizer
        optimizer = optim.AdamW(model.parameters(), lr = lr_gender)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,10])
    
        #loss
        criterion = nn.CrossEntropyLoss()
                    
        best_acc_gender = 0
        best_val_loss_gender = 999
        print(f'Fold {fold+1}\n')
        for epoch in range(num_epochs):
            print(f'epoch: {epoch}\n')
            train_loss_gender = 0
            val_loss_gender = 0
        
            #Training
            model.train()
            iterat = 0
            vsego = len(train_loader)
            for batch in train_loader:
    
                print(f'batch_num: {100*(iterat/vsego)}%\n')
                # Load image batch
                batch_data, batch_gender_label = batch
                batch_data = batch_data.to(device)
                batch_gender_label = batch_gender_label.to(device)
                
                iterat = iterat + 1
                # Clear gradients
                optimizer.zero_grad()
            
                with torch.set_grad_enabled(True):
                    
                    pred_gender = model(batch_data)
                    loss_gender = criterion(pred_gender, batch_gender_label)
            
                    train_loss_gender += loss_gender.detach().item()
                    loss_gender.backward()
                    optimizer.step()
            
            #Validation
            model.eval()
            all_pred_gender = torch.empty(0).to(device)
            for batch in val_loader:
            
                # Load image batch
                batch_data, batch_gender_label = batch
                batch_data = batch_data.to(device)
                batch_gender_label = batch_gender_label.to(device)
                
                with torch.set_grad_enabled(False):
                
                    pred_gender = model(batch_data)
                   
                    loss_gender = criterion(pred_gender, batch_gender_label)
            
                    val_loss_gender += loss_gender.detach().item()
                
                    all_pred_gender = torch.cat((all_pred_gender, 
                            nn.functional.softmax(pred_gender.detach(),dim=1)), 0)
                
        
            train_loss_gender /= len(train_loader)
            val_loss_gender /= len(val_loader)
        
            all_pred_gender = all_pred_gender.cpu().numpy()
            pred_label_gender = list(np.argmax(all_pred_gender,axis=1))
       
        
            acc_gender = accuracy_score(val_gender_label, pred_label_gender)
 
            if val_loss_gender < best_val_loss_gender:
                best_acc_gender=acc_gender
                best_val_loss_gender=val_loss_gender
                torch.save(model.state_dict(), f'models/gender_model{fold}.pth')
            
            all_stat['train_loss'].append(train_loss_gender)
            all_stat['val_loss'].append(val_loss_gender)
            all_stat['val_acc'].append(acc_gender)
            
            print(f'Epoch {epoch} | train loss: {train_loss_gender} | val loss: {val_loss_gender} | accuracy: {round(acc_gender*100, 2)}%')
            scheduler.step()
        
        #INFERENCE
        with torch.no_grad():
            model.load_state_dict(torch.load(f'models/gender_model{fold}.pth'))
            model.eval()
            test_pred_gender = torch.empty(0).to(device)
            for batch in test_loader:
            
                # Load image batch
                batch_data, batch_gender_label = batch
                batch_data = batch_data.to(device)
                batch_gender_label = batch_gender_label.to(device)
            
                with torch.set_grad_enabled(False):
                
                    pred_gender = model(batch_data)
               
                    test_pred_gender = torch.cat((test_pred_gender, 
                            nn.functional.softmax(pred_gender.detach(),dim=1)), 0)
                
            test_pred_gender = test_pred_gender.cpu().numpy()
            pred_label_gender = list(np.argmax(test_pred_gender,axis=1))
        
            acc_gender = accuracy_score(test_gender_label, pred_label_gender)
            all_stat['test_acc'].append(acc_gender)
            all_stat['conf'].append(confusion_matrix(test_gender_label, pred_label_gender, labels=list(range(num_gender_classes))))
            all_stat['conf_norm'].append(confusion_matrix(test_gender_label, pred_label_gender,normalize='true', labels=list(range(num_gender_classes))))
            all_stat['test_pred'].append(pred_label_gender)
            all_stat['test_target'].append(test_gender_label)
        all_accuracy_gender.append(acc_gender)
        all_val_loss_gender.append(best_val_loss_gender)
        print(f'TEST ACCURACY: {round(acc_gender*100,2)}% | Val. Accuracy: {round(best_acc_gender*100,2)}% | Val. Loss.: {best_val_loss_gender}\n')
        
        all_stat_fold.append(all_stat)

    all_accuracy_gender = np.array(all_accuracy_gender)
    all_val_loss_gender = np.array(all_val_loss_gender)

    mean_accuracy_gender = round(all_accuracy_gender.mean()*100, 2)

    print(f'\nOverall Accuracy: {mean_accuracy_gender} p/m')

def train_age(
            kfold,
            batchsize,
            lr_age,
            lr_gender,
            num_epochs,
            p_augment,
            device, num_age_classes, num_gender_classes,
            test_fold, train_fold, random_seed):
    all_accuracy_age = []
    all_val_loss_age = []
    all_stat_fold = []
    
    for fold in range(kfold):
        all_stat = defaultdict(list)
        
        # image paths
        train_data = train_fold[fold]['image_path'].copy().reset_index(drop=True).to_list()
        test_data  = test_fold[fold]['image_path'].copy().reset_index(drop=True).to_list()
    
        #get label
        train_age_label = train_fold[fold]['age'].copy().reset_index(drop=True).to_list()
        train_gender_label = train_fold[fold]['gender'].copy().reset_index(drop=True).to_list()
        test_age_label = test_fold[fold]['age'].copy().reset_index(drop=True).to_list()
        test_gender_label = test_fold[fold]['gender'].copy().reset_index(drop=True).to_list()
   
        #create train-validation stratified split
        sss = StratifiedShuffleSplit(n_splits=10, random_state=random_seed)
    
        #split based on age, more balanced for both age and gender
        train_idx, val_idx = list(sss.split(train_data, train_age_label))[0]
    
        train_idx = list(train_idx)
        val_idx = list(val_idx)
    
        #create dataloader for gender
        train_dataset = AgeDataset('', 
                                          list(np.array(train_data)[train_idx]), 
                                          list(np.array(train_age_label)[train_idx]),
                                          list(np.array(train_gender_label)[train_idx]),
                                          p_augment = p_augment)
        val_dataset   = AgeDataset('', 
                                          list(np.array(train_data)[val_idx]), 
                                          list(np.array(train_age_label)[val_idx]),
                                          list(np.array(train_gender_label)[val_idx]),
                                          validation=True)
        test_dataset = AgeDataset('', 
                                  test_data, 
                                  test_age_label,
                                  test_gender_label, 
                                  validation=True)
    
    
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
    
        val_gender_label = list(np.array(train_gender_label)[val_idx])
        val_age_label = list(np.array(train_age_label)[val_idx])
    
        model = InceptionResnetV1(
                        classify=True,
                        pretrained='vggface2',
                        num_classes=num_age_classes
                                )
        model = model.to(device)
    
        #optimizer
        optimizer = optim.AdamW(model.parameters(), lr = lr_age)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,10])
    
        #loss
        criterion = nn.CrossEntropyLoss()
            
        best_acc_age = 0
        best_val_loss_age = 999

        print(f'Fold {fold+1}\n')
        for epoch in range(num_epochs):
            print(f'epoch: {epoch}\n')
            train_loss_age = 0
            val_loss_age = 0
        
            #Training
            model.train()
            iterat = 0
            vsego = len(train_loader)
            for batch in train_loader:
                print(f'batch_num: {100*(iterat/vsego)}%\n')

                # Load image batch
                batch_data, batch_age_label = batch
                batch_data = batch_data.to(device)
                batch_age_label = batch_age_label.to(device)
                
                # Clear gradients
                optimizer.zero_grad()
            
                with torch.set_grad_enabled(True):
                    pred_age = model(batch_data)
                    loss_age = criterion(pred_age, batch_age_label)
            
                    train_loss_age += loss_age.detach().item()
                    loss_age.backward()
                    optimizer.step()
                
                iterat = iterat + 1
            
            #Validation
            model.eval()
            all_pred_age = torch.empty(0).to(device)
            for batch in val_loader:
            
                # Load image batch
                batch_data, batch_age_label = batch
                batch_data = batch_data.to(device)
                batch_age_label = batch_age_label.to(device)
                
                with torch.set_grad_enabled(False):
                    pred_age = model(batch_data)
                    loss_age = criterion(pred_age, batch_age_label)
                    val_loss_age += loss_age.detach().item()
                    all_pred_age = torch.cat((all_pred_age, 
                            nn.functional.softmax(pred_age.detach(),dim=1)), 0)     
        
            train_loss_age /= len(train_loader)
            val_loss_age /= len(val_loader)
        
            all_pred_age = all_pred_age.cpu().numpy()
            pred_label_age = list(np.argmax(all_pred_age,axis=1))
       
            acc_age = accuracy_score(val_age_label, pred_label_age)
            if acc_age > best_acc_age:
                best_acc_age=acc_age
                best_val_loss_age=val_loss_age
                torch.save(model.state_dict(), f'models/age_model{fold}.pth')
            
            all_stat['train_loss'].append(train_loss_age)
            all_stat['val_loss'].append(val_loss_age)
            all_stat['val_acc'].append(acc_age)
            
            print(f'Epoch {epoch} | train loss: {train_loss_age} | val loss: {val_loss_age} | accuracy: {round(acc_age*100, 2)}%')
            scheduler.step()
        
        #INFERENCE
        with torch.no_grad():
            model.load_state_dict(torch.load(f'models/age_model{fold}.pth'))
            model.eval()
            test_pred_age = torch.empty(0).to(device)
            for batch in test_loader:
            
                # Load image batch
                batch_data, batch_age_label = batch
                batch_data = batch_data.to(device)
                batch_age_label = batch_age_label.to(device)
            
                with torch.set_grad_enabled(False):
                    pred_age = model(batch_data)
                    test_pred_age = torch.cat((test_pred_age, 
                            nn.functional.softmax(pred_age.detach(),dim=1)), 0)
                
        
       
        
            test_pred_age = test_pred_age.cpu().numpy()
            pred_label_age = list(np.argmax(test_pred_age,axis=1))
        
            acc_age = accuracy_score(test_age_label, pred_label_age)
            all_stat['test_acc'].append(acc_age)
            all_stat['conf'].append(confusion_matrix(test_age_label, pred_label_age, labels=list(range(num_age_classes))))
            all_stat['conf_norm'].append(confusion_matrix(test_age_label, pred_label_age,normalize='true', labels=list(range(num_age_classes))))
            all_stat['test_pred'].append(pred_label_age)
            all_stat['test_target'].append(test_age_label)
            
        all_accuracy_age.append(acc_age)
        all_val_loss_age.append(best_val_loss_age)
        print(f'TEST ACCURACY: {round(acc_age*100,2)}% | Val. Accuracy: {round(best_acc_age*100,2)}% | Val. Loss.: {best_val_loss_age}\n')
        
        all_stat_fold.append(all_stat)

    all_accuracy_age = np.array(all_accuracy_age)
    all_val_loss_age = np.array(all_val_loss_age)

    mean_accuracy_age = round(all_accuracy_age.mean()*100, 2)

    print(f'\nOverall Accuracy: {mean_accuracy_age} p/m')

start()