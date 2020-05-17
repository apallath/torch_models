import cv2
import os
import numpy as np

if __name__ == '__main__':
    name = os.listdir('./cell_images')
    np.random.shuffle(name)

    #training and test sets
    train = name[:int(0.80*len(name))]
    test = name[int(0.80*len(name)):]

    #smaller training and test sets, for debugging purposes
    train_small = name[:int(0.10*len(name))]
    test_small = name[int(0.90*len(name)):]

    """training set"""
    os.mkdir('./train')
    images = []
    for i in range(0,len(train)):
        nm = train[i]
        img = cv2.imread('./cell_images/'+nm, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    #normalize data
    images = np.array(images)
    mu = images.mean(axis=0)
    sigma = images.std(axis=0)
    images = (images - mu)/sigma
    #write data
    for i in range(0,len(train)):
        nm = train[i]
        np.save('./train/'+nm.split(".")[0], images[i,:,:])

    """small training set"""
    os.mkdir('./train_small')
    images = []
    for i in range(0,len(train_small)):
        nm = train_small[i]
        img = cv2.imread('./cell_images/'+nm, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    #normalize data
    images = np.array(images)
    mu_small = images.mean(axis=0)
    sigma_small = images.std(axis=0)
    images = (images - mu_small)/sigma_small
    #write data
    for i in range(0, len(train_small)):
        nm = train_small[i]
        np.save('./train_small/'+nm.split(".")[0], images[i,:,:])

    """test set"""
    os.mkdir('./test')
    images = []
    for i in range(0,len(test)):
        nm = test[i]
        img = cv2.imread('./cell_images/'+nm, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    images = (images - mu)/sigma
    #write data
    for i in range(0, len(test)):
        nm = test[i]
        np.save('./test/'+nm.split(".")[0], images[i,:,:])

    """smaller test set"""
    os.mkdir('./test_small')
    images = []
    for i in range(0,len(test_small)):
        nm = test_small[i]
        img = cv2.imread('./cell_images/'+nm, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    images = (images - mu_small)/sigma_small
    #write data
    for i in range(0, len(test_small)):
        nm = test_small[i]
        np.save('./test_small/'+nm.split(".")[0], images[i,:,:])
