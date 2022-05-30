#Import Libraries
import matplotlib.pyplot as plt
import argparse
import torch
import time
import numpy as np
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from PIL import Image
from utils import save_checkpoint, load_checkpoint
import json
import os
import random
from utils import load_checkpoint, load_cat_names

#Defining arg parser
def parse_args():
    #Defining parser
    parser = argparse.ArgumentParser(description="predicting.py")
    #Adding GPU to parser
    parser.add_argument('--gpu', action='store', dest='gpu', default='gpu')
    #Loading checkpoint 
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth', type=str, nargs='*')
    #Specifying the path
    parser.add_argument('--filepath', action='store', dest='filepath', default='flowers/test/1/image_06752.jpg',nargs='*') 
    #Specifying top-k
    parser.add_argument('--top_k', action='store', dest='top_k', type=int ,default='10')
    #Mapping category names
    parser.add_argument('--category_names', action='store', dest='category_names', default='cat_to_name.json')
    
    #Parse args
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
  
    #Using image file path to convert images to PIL 
    img_pil = Image.open(image) 
   
    #Image transform
    tuning = transforms.Compose([transforms.CenterCrop(224), transforms.Resize(256), transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) ])

    #Transforming image for the network
    pil_image_transform = tuning(pil_image)
    #Converting to Numpy array
    numpy_array_image = np.array(pil_image_transform)
    
    return numpy_array_image

def predict(image_path, model, topk=10):
    
    if type(top_k) == type(None):
        top_k = 10
        print('top_K is not not identified')
    
    model.eval()

    #Converting to PyTorch
    image = process_image(image_path)   
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_add_dimension = image_tensor.unsqueeze_(0)
    
    #Converting to linear scale
    probabilities = torch.exp(output)
     #Top 10 results
    top_probabilities = probs.topk(10)[0]
    top_index = probs.topk(10)[1]

    model=model.cpu()
    
    top_probabilities_list = np.array(top_probabilities)[0]
    top_index_list = np.array(top_index[0])
    
    class_to_index = loaded_model.class_to_index

    index_to_class = {x: y for y, x in class_to_index.items()
                       }

    top_classes_list = []
    for index in index_top_list:
        top_classes_list += [index_to_class[index]]
    
    return top_probabilities_list, top_index_list, top_classes_list

##All the above functions are called and executed are in main()  
def main(): 
    #Get keyword Args for training
    args = parse_args()
    #GPU Check
    gpu = args.gpu
    #Loading category names
    with open(args.category_names, 'r') as f: cat_to_name = json.load(f)
    #Loading checkpoint
    model = load_checkpoint(args.checkpoint)
    #Predict top K 
    top_probabilities_list, top_index_list, top_classes_list = predict(image_tensor, model, gpu, category_to_name, args.top_k)
    #Print  probabilities
    print('Probabilities: ', top_probabilities_list)
    print('Classes:       ', top_classes_list)
    print('Predicting has been done successfully')

#Run program       
if __name__ == '__main__':
  main()


