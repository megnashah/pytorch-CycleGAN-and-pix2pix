import os, os.path
import random
import shutil


A_image_directory = 'C:/Users/megna/Documents/DATA/Martensite/reconstructed_data/Outline/'
B_image_directory = 'C:/Users/megna/Documents/DATA/Martensite/reconstructed_data/Packet_ID/'
final_A_directory = 'C:/Users/megna/Documents/DATA/Martensite/current_run/A/'
final_B_directory = 'C:/Users/megna/Documents/DATA/Martensite/current_run/B/'

list_A_images = os.listdir(A_image_directory)
list_B_images = os.listdir(B_image_directory)

num_A_images = len(list_A_images)
num_B_images = len(list_B_images)

if(num_A_images != num_B_images):
    print("the number of images in dir A do not match the number in dir B")
    exit()


list_AB_images = list(zip(list_A_images, list_B_images))

random.shuffle(list_AB_images)

num_test_images = int(0.1*num_A_images)
num_val_images = int(0.1*num_A_images)

index = 0
for imageA, imageB in list_AB_images: 

    if(index < num_test_images): 
        shutil.copy(A_image_directory + imageA, final_A_directory + '/test/' + imageA)
        shutil.copy(B_image_directory + imageB, final_B_directory + '/test/' + imageB)
    elif(index < num_val_images + num_test_images): 
        shutil.copy(A_image_directory + imageA, final_A_directory + '/val/' + imageA)
        shutil.copy(B_image_directory + imageB, final_B_directory + '/val/' + imageB)
    else: 
        shutil.copy(A_image_directory + imageA, final_A_directory + '/train/' + imageA)
        shutil.copy(B_image_directory + imageB, final_B_directory + '/train/' + imageB)

    index += 1

