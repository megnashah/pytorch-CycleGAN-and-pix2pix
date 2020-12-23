import pandas as pd
import os
import numpy
import itertools
from collections import deque
import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
import pygsheets

# This script reads text files from each trial run in our microstructure analyses.
# After we train/test our model, text files are saved containing all options used in the run.
# These options can be found in the project folder 'checkpoints'.
# The text files are read into a pandas dataframe, and later exported to a google sheet via API call.

# NOTE: this code was updated with an assumption that the spreadsheet already had a header row (trial run option attributes).

# TO BE CHANGED: trial_name
trial_name = "trial_12_14_20"
trial_dir = '/home/tom_phelan_ext/gitCode/pix2pix/pytorch-CycleGAN-and-pix2pix/checkpoints/' + trial_name + '/'

# # GOOGLE SHEETS - API CONNECTION via pygsheets package
gc = pygsheets.authorize(service_file='./rxcm-shah-224d456bb1a5.json')
sheet = gc.open_by_key('1bfnht32HCtygzsiq-XjtG8wqwwVRO3LEddEortAdAfc')

# iterate over only text files (train_opt, test_opt)
for filename in os.listdir(trial_dir):
    if filename.endswith(".txt") and filename.startswith(("train", "test")):

        # get current tab name of worksheet & number of rows
        tab_name = str(filename)
        tab_name = tab_name[:-(4)]
        current_ws = sheet.worksheet_by_title(tab_name)
        print("Reading existing ", current_ws, " data...")
        cells = current_ws.get_all_values(include_tailing_empty_rows=False, include_tailing_empty=False, returnas='matrix')
        row_index = len(cells)
        print("Number of rows in ", current_ws, ": ", row_index)
        print("Inserting new data at row index: ", row_index + 1)

        file_path = str(trial_dir + filename)
        attr_array = deque([])
        val_array = deque([])
        val_list = list()

        trial_info = pd.DataFrame()
        # read thru text file of trial/test run options
        with open(file_path, 'r') as readfile:
            last_line = len(readfile.readlines())-1
            # file must be-reopened
            readfile = open(file_path, 'r')

            for line in itertools.islice(readfile, 1, last_line): # omit header and footer
                option = line.split()
                option_attr = option[0][:-(1)] # remove colon from name
                option_val = ''
                if 1 < len(option): # ensure option attribute has a value
                    option_val = option[1]
                print(option_attr + ': ' + option_val)

                # add data to array queues; rotate 'name' to front
                attr_array.append(option_attr)
                val_array.append(option_val)
                if (option_attr == 'name'):
                    attr_array.rotate(1)
                    val_array.rotate(1)

            for value in val_array: val_list.append(value) # convert deque to list
            current_ws.insert_rows(row=row_index, number=1, values=val_list)