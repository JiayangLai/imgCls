# coding=utf-8
import cv2
import os
from os.path import join
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

def appendImgLabel(par_path,cls,imgs,labels):
    print(cls,":")
    if cls=="A":
        label = [1,0,0]
    elif cls=="B":
        label = [0,1,0]
    elif cls=="C":
        label = [0,0,1]

    imgs_cls = os.listdir(join(par_path, cls))

    for img in tqdm(imgs_cls):
        img = cv2.imread(join(par_path, cls, img))
        img = cv2.resize(img, (32, 32))
        img = img.transpose(2,1,0)
        imgs = imgs+[img]
        labels = labels+[label]
    return imgs,labels

if __name__=="__main__":
    par_path = "data"
    clses = ["A","B","C"]
    imgs = []
    labels = []
    for cls in clses:
        imgs, labels = appendImgLabel(par_path,cls, imgs, labels)
    x = np.array(imgs)
    y = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    np.save("x_train.npy", x_train)
    np.save("y_train.npy", y_train)

    np.save("x_test.npy", x_test)
    np.save("y_test.npy", y_test)

