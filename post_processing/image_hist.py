import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import os
import math
from datetime import date
import cv2

# GCP paths will be assigned in code below
# output_folder = ''
# results = '/home/tom_phelan_ext/gitCode/pix2pix/pytorch-CycleGAN-and-pix2pix/results/'
# p2b = '/home/tom_phelan_ext/Documents/microstructure_analysis/packets2blocks/feature_data_FAKE/'
# g2p = '/home/tom_phelan_ext/Documents/microstructure_analysis/grains2packets/feature_data_FAKE/'
# output_p2b = '/home/tom_phelan_ext/Documents/image_histograms/packets2blocks/'
# output_g2p = '/home/tom_phelan_ext/Documents/image_histograms/grains2packets/'

# Megna paths will be assigned in code below
output_folder = ''
results = r'D:\steelGAN\12292020\results\results' + '\\'
p2b = r'D:\steelGAN\12292020\microstructure_analysis\microstructure_analysis\packets2blocks\feature_data_FAKE' + '\\'
g2p = r'D:\steelGAN\12292020\microstructure_analysis\microstructure_analysis\grains2packets\feature_data_FAKE' + '\\'
output_p2b = r'D:\steelGAN\12292020\image_histograms\packets2blocks' + '\\'
output_g2p = r'D:\steelGAN\12292020\image_histograms\grains2packets' + '\\'

# create output directories if do not exist
if not(os.path.exists(output_p2b)): os.makedirs(output_p2b)
if not(os.path.exists(output_g2p)): os.makedirs(output_g2p)

# generate histogram function
def create_histogram(real, fake_arrays, num_fake, trial, labels, output_folder):
    print('Generating histogram for ' + trial + '...')
    print('Number of fake images captured: ', num_fake)
    hist_real, edges = numpy.histogram(real, bins=255, density=True) #cv2.calcHist([real],[0],None,[256],[0,255])
    plt.plot(hist_real, color='k', label='real')

    for index, fake in enumerate(fake_arrays): 
        hist_fake, edges = numpy.histogram(fake, bins=edges, density=True) #cv2.calcHist([fake],[0],None,[256],[0,255])
        plt.plot(hist_fake, label=labels[index])
    plt.legend(loc='upper right')
    plt.xlim(0,255)
    #plt.ylim(0, 0.05)
    plt.xlabel('Intensity')
    #plt.title('Histogram of images in ' + trial)
    plt.savefig(output_folder + trial + '_hist.png')
    print('Histogram saved.')
    print()
    plt.clf()

# read image function using cv2
def read_image(image, path):
    return cv2.imread(path + image, -1)[:, :, 0]
# ---------------------------------------------------------------------------------------- #

trials = os.listdir(results)

# trials = ["trial_12_14_20", 'train_12_28v2_20', 'train_12_30_20'] #packets2blocks smooth 
# labels = ['1', '50', '16'] #packets2blocks smooth 

# trials = ['train_12_21_20'] #packets2blocks sharp
# labels = ['1'] #packets2blocks sharp

# trials= ['trial_12_05_20',  'train_12_28_20', 'train_12_29_20'] #grains2packets smooth
# labels = ['1',  '50', '16'] #grains2packets smooth 

trials = ['train_12_23_20', 'train_12_26_20',] #grains2packets sharp 
labels = ['1', '50'] #grains2packets sharp 

p2b_trials = os.listdir(p2b)
g2p_trials = os.listdir(g2p)
print('p2b trials: ', p2b_trials)
print('g2p trials: ', g2p_trials)
print()

fake_arrays = []
real_array = []

for index, trial in enumerate(trials):
    images_path = results + '/' + trial + '/test_latest/images/'
    # obtain project_type and print to terminal
    if str(trial) in p2b_trials:
        output_folder = output_p2b
        print(str(trial) + ' is a packets2blocks trial.')
    else:
        output_folder = output_g2p
        print(str(trial) + ' is a grains2packets trial.')

    # read thru images of current trial and capture one real, all fake images
    images = os.listdir(results + '/' + trial + '/test_latest/images/')
    
    num_fake = 0
    fake_array = []
    

    # captures all fake images and puts into numpy array
    for image in images:
        if str(image).find('fake_B') > -1: # returns >= 0 if found (index)
            fake_array.append(read_image(str(image), images_path))
            num_fake += 1

    fake_array = numpy.asarray(fake_array)
    fake_arrays.append(fake_array)
    # captures all real 
    for image in images:
        if str(image).find('real_B') > -1:
            real_array.append(read_image(str(image), images_path))
real_array = numpy.asarray(real_array)   
    
create_histogram(real_array, fake_arrays, num_fake, trial, labels, output_folder)
           