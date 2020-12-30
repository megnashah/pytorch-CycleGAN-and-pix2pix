import json
import subprocess
import os
import pandas as pd

def filecount(dir_name):
    return len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])

# TO BE CHANGED: trial_name
trial_name = 'train_12_30_20'


file_data = pd.DataFrame(columns=["image name", "folder", ".csv file"])

# GCP paths for pipeline runs and outputting image data
# csv_file_data = trial_name + "_FAKE.csv"
# csv_file_data_dir = '/home/tom_phelan_ext/Documents/microstructure_analysis/packets2blocks/'
# pipeline_file = '/home/tom_phelan_ext/Documents/dream3d_pipelines/p2b_FAKE_feature_sizes.json'
# image_folder = '../results/' + trial_name + '/test_latest/images/'
# output_csv_folder = csv_file_data_dir + 'feature_data_FAKE/' + trial_name + '/'
# pipeline_runner = '/home/tom_phelan_ext/Programs/DREAM3D/bin/PipelineRunner'

# Megna computer paths for pipeline runs and outputting image data
csv_file_data = trial_name + "_FAKE.csv"
csv_file_data_dir = r'D:\steelGAN\12292020\microstructure_analysis\microstructure_analysis\packets2blocks\\'
pipeline_file = r'C:\Users\shahmn\Documents\CODE\pytorch-CycleGAN-and-pix2pix\dream3d_pipelines\p2b_FAKE_feature_sizes.json'
image_folder = r'D:\steelGAN\12292020\results\results' + '\\' + trial_name + '\\test_latest\\images\\'
output_csv_folder = csv_file_data_dir + 'feature_data_FAKE\\' + trial_name + '\\'
pipeline_runner = r'C:\AdditionalPrograms\DREAM3D-6.5.128-Win64\PipelineRunner.exe'


if not(os.path.exists(output_csv_folder)): os.makedirs(output_csv_folder)

print(output_csv_folder)
total_index = 1

# iterate thru all images in current image folder
imageList = os.listdir(image_folder)
startNumber = 0
numImages = filecount(image_folder)

for i in range(startNumber, startNumber + numImages):
    # ONLY FAKE IMAGES --> will return index of first occurence of 'fake'
    if (str(imageList[i]).find("fake") != -1):
        print(str(imageList[i]))
        # pipeline details, output .csv file
        with open(pipeline_file) as pipeline_json:
            pipeline_json_data = json.load(pipeline_json)
        pipeline_json_data['00']['FileName'] = image_folder + imageList[i]
        pipeline_json_data['40']['FeatureDataFile'] = output_csv_folder + str(total_index) + '.csv'

        with open(pipeline_file, 'w') as pipeline_json:
            pipeline_json.write(json.dumps(pipeline_json_data, indent=4))

        process_call = pipeline_runner + ' -p' + ' ' + pipeline_file

        print('*********************************')
        print('Running permutation {} of {}'.format(i, numImages))
        print('*********************************')

        subprocess.call(process_call, shell=True)
    
        # add to pandas dataFrame (.csv file later)
        file_data_tuple = pd.DataFrame({"image name": imageList[i], "folder": "images", ".csv file": str(total_index) + ".csv"}, index=[total_index])
        print(file_data_tuple)
        file_data = pd.concat([file_data, file_data_tuple])

        total_index += 1

print(file_data.head())
# parse to .csv file with given parameters
file_data.to_csv(csv_file_data_dir + csv_file_data)
