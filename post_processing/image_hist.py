import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import os
import math
from datetime import date
import cv2

# will be assigned in code below
output_folder = ''
results = '/home/tom_phelan_ext/gitCode/pix2pix/pytorch-CycleGAN-and-pix2pix/results/'
p2b = '/home/tom_phelan_ext/Documents/microstructure_analysis/packets2blocks/feature_data_FAKE/'
g2p = '/home/tom_phelan_ext/Documents/microstructure_analysis/grains2packets/feature_data_FAKE/'
output_p2b = '/home/tom_phelan_ext/Documents/image_histograms/packets2blocks/'
output_g2p = '/home/tom_phelan_ext/Documents/image_histograms/grains2packets/'

# create output directories if do not exist
if not(os.path.exists(output_p2b)): os.makedirs(output_p2b)
if not(os.path.exists(output_g2p)): os.makedirs(output_g2p)

# generate histogram function
def create_histogram(real, real_name, fake, num_fake, trial, output_folder):
    print('Generating histogram for ' + trial + '...')
    print('Real image selected: ' + real_name)   
    print('Number of fake images captured: ', num_fake)
    hist_real = cv2.calcHist([real],[0],None,[256],[0,255])
    plt.plot(hist_real, color='k', label='real')
    hist_fake = cv2.calcHist([fake],[0],None,[256],[0,255])
    plt.plot(hist_fake, color='r', label='fake')
    plt.legend(loc='upper right')
    plt.xlim(0,255)
    plt.xlabel('Intensity')
    plt.title('Histogram of images in ' + trial)
    plt.savefig(output_folder + trial + '_hist.png')
    print('Histogram saved.')
    print()
    plt.clf()

# read image function using cv2
def read_image(image, path):
    return cv2.imread(path + image, -1)
# ---------------------------------------------------------------------------------------- #

trials = os.listdir(results)
p2b_trials = os.listdir(p2b)
g2p_trials = os.listdir(g2p)
print('p2b trials: ', p2b_trials)
print('g2p trials: ', g2p_trials)
print()

for trial in trials:
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
            fake_array = numpy.concatenate(read_image(str(image), images_path))
            num_fake += 1

    # captures one 
    for image in images:
        if str(image).find('real_B') > -1:
            real_name = str(image)
            real = read_image(real_name, images_path)
            create_histogram(real, real_name, fake_array, num_fake, trial, output_folder)
            break # only takes one real image for histogram
