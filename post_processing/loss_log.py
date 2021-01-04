import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import itertools
import os

# TO BE CHANGED --> trial_name, num_epochs, num_xticks
trial_name = '12_28v2'
project = 'packets2blocks'

#GCP directories
# loss_file = '../checkpoints/' + trial_name + '/loss_log.txt'
# graphs_folder = '/home/tom_phelan_ext/Documents/graphs/' + project + '/' + trial_name + '/'

#Megna Directories
loss_directory = r'D:\steelGAN\12292020\loss'
loss_file = loss_directory + '\\' + trial_name + '_loss.txt'
graphs_folder = loss_directory + '\\plots\\' + trial_name

num_epochs = 1100
num_xticks = 185

if not(os.path.exists(graphs_folder)): os.makedirs(graphs_folder)
rows_list = []

def get_min_and_max(final_min, final_max, data):
    loss_data = [float(i) for i in data.values]
    current_min = min(loss_data)
    current_max = max(loss_data)
    if(final_min > current_min): final_min = current_min
    if(final_max < current_max): final_max = current_max
    return final_min, final_max, loss_data

# LOSS LOG DATA ----------------------------------------------------------------------------------- #

with open(loss_file, 'r') as readfile:
    last_line = len(readfile.readlines())
    # file must be re-opened
    readfile = open(loss_file, 'r')
    for line in itertools.islice(readfile, 1, last_line): # omits training model header info
        ep = line.split()
        # NOTE: hard-coded the indices for appending data to dataframe (element indices are known)
        ep_name = ep[1][:-(1)] # omit trailing comma
        rows_list.append({'epoch': ep_name, 'G_GAN': ep[9], 'G_L1': ep[11], 'D_real': ep[13], 'D_fake': ep[15]})

epoch_data = pd.DataFrame(rows_list)
epoch_data.columns = ['epoch', 'G_GAN', 'G_L1', 'D_real', 'D_fake']
print(epoch_data)
print()

epochs = epoch_data['epoch'].values
loss_array = [[],[],[],[]]
print("Epochs: ", epochs)
print()

# LOSS CURVES ------------------------------------------------------------------------------------- #


# NOTE: we do not use min and max return values in the first loop. calculate min/max for each plot.
# NOTE: this is because we want unique min/max values for each loss curve; calling function will overwrite them.

ymin = 10000
ymax = 0
# plotting epoch vs. each loss, individually
print("Generating EquivalentDiameters individual loss plots...")
print()
colors = ['b', 'g', 'm', 'y']
for i in range(1, len(epoch_data.columns)):
    attr = epoch_data.columns[i]
    print("Generating " + attr + " loss plot...")
    print(attr + ":", epoch_data[attr].values)
    x, y, loss_list = get_min_and_max(ymin, ymax, epoch_data[attr]) # x,y garbage variables
    ymin = min(loss_list)
    ymax = max(loss_list)
    print(attr + " ymin: ", ymin)
    print(attr + " ymax: ", ymax)
    plt.xlim(1, num_epochs) # num epochs
    plt.ylim(ymin, ymax) # range of losses
    plt.xticks(numpy.arange(1, num_epochs, num_xticks)) # scale x-axis ticks
    plt.scatter(epochs, loss_list, s=5, c=colors[i-1])
    plt.title(attr + " Loss for " + trial_name + " Images")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(graphs_folder + attr + "_loss.png")
    plt.clf()
    print(attr + " loss plot completed and saved.")
    print()


# plotting epoch vs. generator loss, grouped plot
print("Generating Generator grouped loss plot...")
print()
ymin = 10000
ymax = 0
alpha = 1
colors = ['b', 'g']
for i in range(1, 3):  #len(epoch_data.columns)
    attr = epoch_data.columns[i]
    print(attr, ':', epoch_data[attr].values)
    ymin, ymax, loss_list = get_min_and_max(ymin, ymax, epoch_data[attr])
    print(attr + " ymin:", ymin)
    print(attr + " ymax:", ymax)
    print()
    plt.scatter(epochs, loss_list, alpha=alpha, label=attr, s=5, c=colors[i-1])
    alpha -= 0.2

# check to see if function returns correct vals
print("final ymin: ", ymin)
print("final ymax: ", ymax)

plt.xlim(1, num_epochs) # num epochs
plt.ylim(ymin, ymax) # range of losses
plt.title("Generator Loss")
plt.xticks(numpy.arange(1, num_epochs, num_xticks)) # scale x-axis ticks
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(graphs_folder + "G_loss.png")
plt.clf()

print("Generating Discriminator grouped loss plot...")
print()
ymin = 10000
ymax = 0
alpha = 1
colors = ['m', 'y']
for i in range(3, 5):  #len(epoch_data.columns)
    attr = epoch_data.columns[i]
    print(attr, ':', epoch_data[attr].values)
    ymin, ymax, loss_list = get_min_and_max(ymin, ymax, epoch_data[attr])
    print(attr + " ymin:", ymin)
    print(attr + " ymax:", ymax)
    print()
    plt.scatter(epochs, loss_list, alpha=alpha, label=attr, s=5, c=colors[i-3])
    #alpha -= 0.2

# check to see if function returns correct vals
print("final ymin: ", ymin)
print("final ymax: ", ymax)

plt.xlim(1, num_epochs) # num epochs
plt.ylim(ymin, 2.0) # range of losses
plt.title("Discriminator Loss")
plt.xticks(numpy.arange(1, num_epochs, num_xticks)) # scale x-axis ticks
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(graphs_folder + "D_loss.png")
plt.clf()








print("Grouped loss plot saved.")
print()
print("Program completed.")