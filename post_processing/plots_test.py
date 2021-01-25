import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import os
import math
import pickle
from datetime import date


# TO BE CHANGED --> list of fake feature data to be analyzed
# trial_list = ["trial_12_14_20", 'train_12_21_20', 'train_12_28v2_20', 'train_12_30_20'] #packets2blocks
# label_list = ['B1, smooth', 'B1, sharp', 'B50, smooth', 'B16, smooth'] #packets2blocks
trial_list = ['trial_12_05_20', 'train_12_23_20', 'train_12_26_20', 'train_12_28_20', 'train_12_29_20'] #grains2packets
label_list = ['B1, smooth', 'B1, sharp', 'B50, sharp', 'B50, smooth', 'B16, smooth'] #grains2packets
#project = 'packets2blocks'
project = 'grains2packets'
# these will be 3D arrays of all fake data collected
all_FAKE = list(range(len(trial_list)))
log_all_FAKE = list(range(len(trial_list)))

#GCP PATHS 
# # directory paths for feature data, both real and fake
# feature_data_dir = "/home/tom_phelan_ext/Documents/microstructure_analysis/" + project + "/feature_data/"
# feature_data = os.listdir(feature_data_dir)
# feature_data_FAKE_dir = "/home/tom_phelan_ext/Documents/microstructure_analysis/" + project + "/feature_data_FAKE/"

# # directory path for output graphs/plots; outputs in new latest trial graph folder
# graphs_folder = "/home/tom_phelan_ext/Documents/graphs/" + project + "/" + trial_list[len(trial_list) - 1] + "/"
# if (not(os.path.exists(graphs_folder))): os.makedirs(graphs_folder)

#Megna PATHS 
# directory paths for feature data, both real and fake
feature_data_dir = r'D:\steelGAN\12292020\microstructure_analysis\microstructure_analysis' + '\\' + project + "\\feature_data\\"
feature_data = os.listdir(feature_data_dir)
feature_data_FAKE_dir = r'D:\steelGAN\12292020\microstructure_analysis\microstructure_analysis' + '\\' + project + "\\feature_data_FAKE\\"

# directory path for output graphs/plots; outputs in new latest trial graph folder
graphs_folder = r'D:\steelGAN\12292020\microstructure_analysis\microstructure_analysis' + '\\' + project + "\\plots\\" 
if (not(os.path.exists(graphs_folder))): os.makedirs(graphs_folder)

# arrays of attributes to be normalized and log normalized
attributes = ['AspectRatios_0', 'AxisEulerAngles_0','Neighborhoods']
log_attributes = ['AxisLengths_0', 'AxisLengths_1', 'EquivalentDiameters', 'NumNeighbors']

def build_attr_array(image, attr, log_bool):
    new_data = image[attr].values
    if attr == 'Neighborhoods':
        # DEBUGGING
        if (max(new_data) >= 580): print("csv file with >=580 neighborhoods: " + str(csv_file))

    if (attr == 'AxisEulerAngles_0'):
        # convert radians to degrees.
        i = 0
        for i in range(len(new_data)):
            new_data[i] = math.degrees(new_data[i])
    
    # checks whether attribute needs log distr.
    if (log_bool):
        new_data = numpy.log(numpy.asarray(new_data))
    # return new_data array, to be appended to its respective attr. array
    return new_data

def create_histogram(real_attr_arr, fake, attr, attr_index, num_bins, log_bool):
    if attr == 'Neighborhoods' or attr == 'NumNeighbors':
        num_bins=10
    # alpha determines transparency, density normalizes the dataset
    #plt.hist(real_attr_arr, bins=num_bins, label='real', density=True, alpha=0.5, edgecolor='blue')
    y, binEdges = numpy.histogram(real_attr_arr, bins=num_bins, density = True)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1]) 
    plt.plot(bincenters, y, label='real', linestyle='dashed', linewidth=4)
    for i in range(0,len(fake)):
        date = label_list[i] #trial_list[i][6:-3] # truncate trial name down to MM-DD for legend labeling
        fake_label = 'fake (' + date + ')'
        sub_arr = fake[i]
        # i = index of current fake dataset
        #plt.hist(sub_arr[attr_index], bins=num_bins, label=fake_label, density=True, alpha=0.5)
        y, binEdges = numpy.histogram(sub_arr[attr_index], bins=binEdges, density=True)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1]) 
        plt.plot(bincenters, y, '-', label=fake_label)
    
    plt.ylabel("Number of Features, normalized")
    plt.legend(loc='upper right')
    # labeling for histogram and image (normal or log normal)
    if (log_bool):
        plt.xlabel('ln(' + attr + ')')
        plt.title('ln(' + attr + ')')
        plt.savefig(graphs_folder + attr + "_log_HIST.png")
    else:
        plt.xlabel(attr)
        plt.title(attr)
        plt.savefig(graphs_folder + attr + "_HIST.png")
    plt.clf()

def read_images(image, data_arr, log_data_arr):
    # NORMAL DATA
    sub_arr = 0
    for attr in attributes:
        data_arr[sub_arr] = numpy.concatenate([data_arr[sub_arr], build_attr_array(image=image, attr=attr, log_bool=False)], axis=0)
        sub_arr += 1

    # LOG DATA
    sub_arr = 0
    for attr in log_attributes:
        log_data_arr[sub_arr] = numpy.concatenate([log_data_arr[sub_arr], build_attr_array(image=image, attr=attr, log_bool=True)], axis=0)
        sub_arr += 1

    # return arrays of image attributes
    return data_arr, log_data_arr

def get_min_and_max(final_min, final_max, data):
    data = [float(i) for i in data]
    current_min = min(data)
    current_max = max(data)
    if(final_min > current_min): final_min = current_min
    if(final_max < current_max): final_max = current_max
    return final_min, final_max

# NOTE: To gather all attribute data without having to reread image .csv files numerous times, we will use 2-D arrays/lists.
# NOTE: Each index of the total image data array will be a sub-array that represents a certain image attribute.

# REAL IMAGE DATA --------------------------------------------------------------------------------- #
data_real = [[],[],[]]
log_data_real = [[],[],[],[]]

print("Processing real image data for " + project + "...")
for csv_file in feature_data:
    image = pd.read_csv(feature_data_dir + csv_file, skiprows=0, header=1)
    data_real, log_data_real = read_images(image, data_real, log_data_real)
    
print(project + " real image data collected.")
print()

# FAKE IMAGE DATA --------------------------------------------------------------------------------- #
i = 0
for trial in trial_list:
    # selects a trial run of fake data to be analyzed
    feature_data_FAKE = feature_data_FAKE_dir + trial + '/'
    print("Processing " + str(trial) + " fake data...")

    data_FAKE = [[],[],[]]
    log_data_FAKE = [[],[],[],[]]

    for csv_file in os.listdir(feature_data_FAKE):
        image = pd.read_csv(feature_data_FAKE + csv_file, skiprows=0, header=1)
        read_images(image, data_FAKE, log_data_FAKE)

    # add fake feature data from current trial to final arrays (all)
    all_FAKE[i] = data_FAKE
    log_all_FAKE[i] = log_data_FAKE
    i += 1
    print(str(trial) + " fake image data collected.")
    print()

# CSV FILE STATS ---------------------------------------------------------------------------------- #
i = 0



for attr in log_attributes:
    print("Generating data for " + attr + " data...")
    stats = []
    stats.append([numpy.mean(log_data_real[i]), numpy.mean(log_all_FAKE[0][i]), numpy.mean(log_all_FAKE[1][i]), numpy.mean(log_all_FAKE[1][i]), numpy.mean(log_all_FAKE[3][i]), numpy.mean(log_all_FAKE[4][i])])
    stats.append([numpy.median(log_data_real[i]), numpy.median(log_all_FAKE[0][i]), numpy.median(log_all_FAKE[1][i]), numpy.median(log_all_FAKE[1][i]), numpy.median(log_all_FAKE[3][i]), numpy.median(log_all_FAKE[4][i])])
    stats.append([numpy.min(log_data_real[i]), numpy.min(log_all_FAKE[0][i]), numpy.min(log_all_FAKE[1][i]), numpy.min(log_all_FAKE[1][i]), numpy.min(log_all_FAKE[3][i]), numpy.min(log_all_FAKE[4][i])])
    stats.append([numpy.max(log_data_real[i]), numpy.max(log_all_FAKE[0][i]), numpy.max(log_all_FAKE[1][i]), numpy.max(log_all_FAKE[1][i]), numpy.max(log_all_FAKE[3][i]), numpy.max(log_all_FAKE[4][i])])
    stats.append([numpy.std(log_data_real[i]), numpy.std(log_all_FAKE[0][i]), numpy.std(log_all_FAKE[1][i]), numpy.std(log_all_FAKE[1][i]), numpy.std(log_all_FAKE[3][i]), numpy.std(log_all_FAKE[4][i])])
    
    i += 1
    
    df = pd.DataFrame(stats, columns = ['real'] + label_list, index = ['mean', 'median', 'min', 'max', 'st dev'])
    df.to_csv(r"D:\steelGAN\12292020\microstructure_analysis\microstructure_analysis\stats\grains2packets" + "\\" + attr + ".csv")

i=0
for attr in attributes:
    print("Generating data for " + attr + " data...")
    stats = []
    stats.append([numpy.mean(data_real[i]), numpy.mean(all_FAKE[0][i]), numpy.mean(all_FAKE[1][i]), numpy.mean(all_FAKE[1][i]), numpy.mean(all_FAKE[3][i]), numpy.mean(all_FAKE[4][i])])
    stats.append([numpy.median(data_real[i]), numpy.median(all_FAKE[0][i]), numpy.median(all_FAKE[1][i]), numpy.median(all_FAKE[1][i]), numpy.median(all_FAKE[3][i]), numpy.median(all_FAKE[4][i])])
    stats.append([numpy.min(data_real[i]), numpy.min(all_FAKE[0][i]), numpy.min(all_FAKE[1][i]), numpy.min(all_FAKE[1][i]), numpy.min(all_FAKE[3][i]), numpy.min(all_FAKE[4][i])])
    stats.append([numpy.max(data_real[i]), numpy.max(all_FAKE[0][i]), numpy.max(all_FAKE[1][i]), numpy.max(all_FAKE[1][i]), numpy.max(all_FAKE[3][i]), numpy.max(all_FAKE[4][i])])
    stats.append([numpy.std(data_real[i]), numpy.std(all_FAKE[0][i]), numpy.std(all_FAKE[1][i]), numpy.std(all_FAKE[1][i]), numpy.std(all_FAKE[3][i]), numpy.std(all_FAKE[4][i])])
    
    i += 1
    
    df = pd.DataFrame(stats, columns = ['real'] + label_list, index = ['mean', 'median', 'min', 'max', 'st dev'])
    df.to_csv(r"D:\steelGAN\12292020\microstructure_analysis\microstructure_analysis\stats\grains2packets" + "\\" + attr + ".csv")

print("lk")



# HISTOGRAMS -------------------------------------------------------------------------------------- #
i = 0
for attr in attributes:
    print("Generating histogram for " + attr + " data...")
    create_histogram(real_attr_arr=data_real[i], fake=all_FAKE, attr=attr, attr_index=i, num_bins=25, log_bool=False)
    i += 1
    print(attr + " histogram saved.")
    print()
i = 0
for attr in log_attributes:
    print("Generating histogram for " + attr + " data...")
    create_histogram(real_attr_arr=log_data_real[i], fake=log_all_FAKE, attr=attr, attr_index=i, num_bins=25, log_bool=True)
    i += 1
    print(attr + " histogram saved.")
    print()

# SCATTER PLOTS ----------------------------------------------------------------------------------- #

# NOTE: in the log arrays, index 2 is EquivalentDiameters. This will be the x-axis on scatter plots.
# plotting all attr in attributes array against EquivalentDiameters
i = 0
xmin = 10000
xmax = 0
for attr in attributes:
    if attr == 'Neighborhoods' or attr == 'NumNeighbors':
        print("Generating Equivalent Diameters vs. " + attr + " scatter plot...")
        xmin, xmax = get_min_and_max(xmin, xmax, data_real[i])
        plt.scatter(log_data_real[2], data_real[i], label='real', alpha=0.5)
        for j in range(len(all_FAKE)):
            # plot each set of fake data
            current_fake = all_FAKE[j]
            temp = log_all_FAKE[j]
            fake_diam = temp[2] # equivalent diameters of current fake dataset
            date = label_list[j] #trial_list[j][6:-3] # truncate trial name down to MM-DD for legend labeling
            fake_label = 'fake (' + date + ')'
            plt.scatter(fake_diam, current_fake[i], label=fake_label, alpha=0.25)
            # check min and max again
            xmin, xmax = get_min_and_max(xmin, xmax, current_fake[i])
        
        print(attr + " xmin: ", xmin)
        print(attr + " xmax: ", xmax)
        # plot image layout and save figure
        plt.title("Relationship in " + attr + " and EquivalentDiameters")
        plt.xlabel('Equivalent Diameters (log)')
        plt.ylabel(attr)
        plt.ylim(xmin, xmax) # sclae x-axis to better fit the data
        plt.legend(loc='upper left')
        plt.savefig(graphs_folder + attr + "_diameters_scatter.png")
        plt.clf()
        print(attr + " scatter plot saved.")
        print()

        #reset xmin and xmax for next attribute
        xmin = 10000
        xmax = 0
    i += 1

print("Program completed.")