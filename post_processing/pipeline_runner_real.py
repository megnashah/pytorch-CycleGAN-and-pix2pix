import json
import subprocess
import os
import pandas as pd

def filecount(dir_name):
    return len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])

# TO BE CHANGED (ONLY for new REAL image batches): csv_file_data
csv_file_data = "trial_12_05_20.csv"

csv_file_data_dir = '/home/tom_phelan_ext/Documents/microstructure_analysis/grains2packets/'
file_data = pd.DataFrame(columns=["image name", "folder", ".csv file"])

# GCP paths for pipeline runs and outputting image data
pipeline_file = '/home/tom_phelan_ext/Documents/dream3d_pipelines/find_feature_sizes.json'
current_run = '/home/tom_phelan_ext/gitCode/pix2pix/pytorch-CycleGAN-and-pix2pix/datasets/current_run/B/'
output_csv_folder = csv_file_data_dir + 'feature_data/'
pipeline_runner = '/home/tom_phelan_ext/Programs/DREAM3D/bin/PipelineRunner'

# subdirs are those listed within image_folder
subdirs = os.listdir(current_run)
print(subdirs)

total_index = 1

for subdir in subdirs:
    # create path for folders: test, train, val
    image_folder = os.path.join(current_run, subdir) + "/"
    print("Image folder path: ", image_folder)
    numImages = filecount(image_folder)
    print("Number of images in ", subdir, ": ", numImages)

    # iterate thru all images in current image folder
    imageList = os.listdir(image_folder)
    startNumber = 0

    for i in range(startNumber, startNumber + numImages):
        # pipeline details, output .csv file
        with open(pipeline_file) as pipeline_json:
            pipeline_json_data = json.load(pipeline_json)
        pipeline_json_data['00']['FileName'] = image_folder + imageList[i]
        pipeline_json_data['13']['FeatureDataFile'] = output_csv_folder + str(total_index) + '.csv'

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
file_data.to_csv(csv_file_data_dir + csv_file_data)
