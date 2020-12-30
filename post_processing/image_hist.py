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
def create_histogram(real, fake, images, trial, output_folder):
    real_image = cv2.imread(images + real, -1)
    fake_image = cv2.imread(images + fake, -1)
    hist_fake = cv2.calcHist([fake_image],[0],None,[256],[0,255])
    plt.plot(hist_fake)
    hist_real = cv2.calcHist([real_image],[0],None,[256],[0,255])
    plt.plot(hist_real,color='r')
    plt.legend(loc='upper right')
    plt.xlim(0,255)
    plt.xlabel('Intensity')
    plt.title('Histogram of images in ' + trial)
    plt.savefig(output_folder + trial + '_hist.png')
    plt.clf()

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

    # read thru images of current trial and capture pair of real & fake
    images = os.listdir(results + '/' + trial + '/test_latest/images/')
    for image in images:
        if str(image).find('fake_B') > -1: # returns >= 0 if found (index)
            fake = str(image)
            real = str(image)[:-10] + 'real_B.png'
            print('fake: ' + fake)
            print('real: ' + real)
            print('Generating histogram for ' + trial + '...')
            create_histogram(real, fake, images_path, trial, output_folder)
            print('Histogram saved.')
            print()
            break
