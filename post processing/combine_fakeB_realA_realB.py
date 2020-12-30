import cv2 
import numpy as np 
import os 

top_directory = r'D:\steelGAN\12292020\results\results' 
folders = ['train_12_21_20', 'train_12_23_20', 'train_12_26_20'] #os.listdir(top_directory)

for folder in folders: 

    directory = top_directory + '\\' + folder  + r'\test_latest\images'
    out_directory = os.path.dirname(directory) + '\\' + 'combined_images\\'
    if (not(os.path.exists(out_directory))): os.makedirs(out_directory)


    files = os.listdir(directory)


    for i in range (0, len(files), 3): 
        fake_B = cv2.imread(directory + '\\' + files[i], -1)[:, :, 0]
        real_A = cv2.imread(directory + '\\' +  files[i+1], -1)[:, :, 0]
        real_B = cv2.imread(directory + '\\' +  files[i+2], -1)[:, :, 0]
        total = np.concatenate((fake_B, real_A, real_B), axis = 1)
        out_file_name = out_directory + files[i].split('f')[0] + '.png'

        cv2.imwrite(out_file_name, total)
