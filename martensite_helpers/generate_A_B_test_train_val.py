import os, os.path
import random
import shutil

source_dir = '/home/tom_phelan_ext/gitCode/pix2pix/pytorch-CycleGAN-and-pix2pix//datasets/raw_data/'
dest_dir = '/home/tom_phelan_ext/gitCode/pix2pix/pytorch-CycleGAN-and-pix2pix/datasets/packets2blocks/'
A_image_directory = source_dir + 'Packet_ID/'
B_image_directory = source_dir + 'Block_ID/'
final_A_directory = dest_dir + 'A/'
final_B_directory = dest_dir + 'B/'

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

