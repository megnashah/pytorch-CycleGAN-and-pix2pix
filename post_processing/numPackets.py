import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import os
import math
from datetime import date

# TO BE CHANGE --> trial
trial = 'trial_12_05_20'

# numPackets directories, both original and cropped, and output folder
orig_imgs_dir = '/home/tom_phelan_ext/Documents/microstructure_analysis/grains2packets/numPackets/' + trial +'/'
cropped_imgs_dir = '/home/tom_phelan_ext/Documents/microstructure_analysis/grains2packets/numPackets_cropped/' + trial + '_cropped/'
output_folder = '/home/tom_phelan_ext/Documents/microstructure_analysis/grains2packets/numPackets_hists/'

def read_csv_files(imgs_dir, img_type):
    # append each image's numPackets data to array
    data = []
    count = 0
    print("Reading csv files of " + img_type + " images...")
    print()
    csv_images = os.listdir(imgs_dir)
    for csv in csv_images:
        csv_data = numpy.genfromtxt(fname=imgs_dir + str(csv), skip_header=2)
        avg = numpy.mean(csv_data)
        if (avg > 150): count += 1
        data.append(avg)
    print("numPackets data for " + img_type + " images collected.")
    print("avg numPackets > 150: ", count)
    print()
    return numpy.array(data)

def create_histogram(array, img_type, log_bool):
    # generate histograms
    if (log_bool): numpy.log(array)
    plt.hist(array, linestyle='dashed', linewidth=4)
    plt.xlabel("numPackets Distribution per Grain")
    plt.ylabel("Frequency")
    # labeling for histogram and image (normal or log normal)
    if (log_bool):
        plt.title("numPackets for " + trial + ", log (" + img_type + ")")
        plt.savefig(output_folder + img_type + "_numPackets_LOG_hist.png")
        print("numPackets log histogram for " + img_type + " images saved.")
    else:
        plt.title("numPackets for " + trial + " (" + img_type + ")")
        plt.savefig(output_folder + img_type + "_numPackets_hist.png")
        print("numPackets histogram for " + img_type + " images saved.")
    plt.clf()

# creates numPacket arrays, original and cropped
avg_numPackets = read_csv_files(imgs_dir=orig_imgs_dir, img_type="original")
avg_numPackets_cropped = read_csv_files(imgs_dir=cropped_imgs_dir, img_type="cropped")
# generate histograms for both original and cropped image data, log histograms too
print("Generating numPackets histograms...")
print()
create_histogram(array=avg_numPackets, img_type="original", log_bool=False)
create_histogram(array=avg_numPackets, img_type="original", log_bool=True)
create_histogram(array=avg_numPackets_cropped, img_type="cropped", log_bool=False)
create_histogram(array=avg_numPackets_cropped, img_type="cropped", log_bool=True)
print()