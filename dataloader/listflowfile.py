import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):


 all_left_img=[]
 all_right_img=[]
 all_left_disp = []
 test_left_img=[]
 test_right_img=[]
 test_left_disp = []


 for im in range(10):
  if is_image_file(filepath+'/TL'+str(im)+'.png'):
   all_left_img.append(filepath+'/TL'+str(im)+'.png')
   all_left_disp.append(filepath+'/TLD'+str(im)+'.pfm')
  if is_image_file(filepath+'/TR'+str(im)+'.png'):
   all_right_img.append(filepath+'/TR'+str(im)+'.png')
  
 #print(all_right_img)
 #print(all_left_img)
 #print(all_left_disp)
 #exit()


 return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


