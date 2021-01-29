import json
import subprocess
import os
import pandas as pd

def filecount(dir_name):
    return len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])

# TO BE CHANGED --> trial, only for new REAL image batches
trial = 'trial_12_05_20'
csv_file_data = trial + '_cropped.csv'

numPackets_folder = '/home/tom_phelan_ext/Documents/microstructure_analysis/grains2packets/numPackets_cropped/'
file_data = pd.DataFrame(columns=["image name", "folder", ".csv file"])

# paths for pipeline runs and outputting image data; folder B in packets2blocks is the real block images
pipeline_file = '/home/tom_phelan_ext/gitCode/pix2pix/pytorch-CycleGAN-and-pix2pix/dream3d_pipelines/g2p_analysis_cropped.json'
grains = '/home/tom_phelan_ext/gitCode/pix2pix/pytorch-CycleGAN-and-pix2pix/datasets/current_run/A/'
packets = '/home/tom_phelan_ext/gitCode/pix2pix/pytorch-CycleGAN-and-pix2pix/datasets/current_run/B/'
output_csv_folder = numPackets_folder + trial + '_cropped/'
pipeline_runner = '/home/tom_phelan_ext/Programs/DREAM3D/bin/PipelineRunner'

# creates trial directory if new data
if (not(os.path.exists(output_csv_folder))): os.makedirs(output_csv_folder)

# subdirs are those listed within image_folder
subdirs = os.listdir(grains)
print(subdirs)

total_index = 1
for subdir in subdirs:
    # create path for folders: test, train, val
    grain_image_folder = os.path.join(grains, subdir) + "/"
    packet_image_folder = os.path.join(packets, subdir) + "/"
    numImages = filecount(grain_image_folder)
    print("Number of grain images in ", subdir, ": ", numImages)
    print("Number of packet images in ", subdir, ": ", filecount(packet_image_folder))

    # iterate thru all images in current image folder
    imageList = os.listdir(grain_image_folder)
    startNumber = 0

    for i in range(startNumber, startNumber + numImages):
        # pipeline details, output .csv file
        with open(pipeline_file) as pipeline_json:
            pipeline_json_data = json.load(pipeline_json)
        pipeline_json_data['00']['FileName'] = grain_image_folder + imageList[i]
        pipeline_json_data['14']['OutputFilePath'] = output_csv_folder + str(total_index) + '.csv'

        with open(pipeline_file, 'w') as pipeline_json:
            pipeline_json.write(json.dumps(pipeline_json_data, indent=4))

        process_call = pipeline_runner + ' -p' + ' ' + pipeline_file

        print('*********************************')
        print('Running permutation {} of {}'.format(i, numImages))
        print('*********************************')

        subprocess.call(process_call, shell=True)
        
        # add to pandas dataFrame (.csv file later)
        file_data_tuple = pd.DataFrame({"image name": imageList[i], "folder": subdir, ".csv file": str(total_index) + ".csv"}, index=[total_index])
        print(file_data_tuple)
        file_data = pd.concat([file_data, file_data_tuple])

        total_index += 1

print(file_data.head())
# parse to .csv file with given parameters
file_data.to_csv(numPackets_folder + csv_file_data)
