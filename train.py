#Import libraries
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


#Define arg parser
def parse_args():
    #Define parser
    parser = argparse.ArgumentParser(description="training.py")
    
    parser.add_argument('--data_dir', action='store', default='./flowers/',
                        help='Enter learning rate for training data')
    
    #Architecture selection addition to parser
    parser.add_argument('--arch', dest='arch', action='store', default='vgg11', choices=['vgg11', 'alexnet'])
    
    #Adding GPU to parser
    parser.add_argument('--gpu', action='store', default='gpu')
   
    #Adding hyperparameter tuning to parser
    parser.add_argument('--epochs', dest='epochs', action='store', default='2')
    parser.add_argument('--learning_rate', dest='learning_rate', action='store', default='0.05')
    parser.add_argument('--hidden_units', dest='hidden_units', action='store', default='150')
    
    #Adding checkpoint to parser
    parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth')
    
    #Parse args
    return parser.parse_args()


#Train transformations on a dataset
def train(model, criterion, optimizer, dataloaders, epochs, gpu):
    steps = 0
    print_every = 50
    
    for e in range(epochs):
        running_loss = 0
        
        for ii, (inputs_train, labels_train) in enumerate(dataloaders[0]): 
            steps += 1 
            #Select CUDA processing if it's supported by the environment
            if torch.cuda.is_available(): 
                model.cuda()
                inputs_train, labels_train = inputs_train.to('cuda'), labels_train.to('cuda') 
                print('Training GPU')
            #CPU selection
            else:
                model.cpu() 
                print('Training CPU')
                
            #Zero parameter gradients    
            optimizer.zero_grad()
            #Train forward and backward passes
            outputs = model.forward(inputs_train)
            loss = criterion(outputs, labels_train)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            #Model validation
            if steps % print_every == 0:
                model.eval()
                validation_loss = 0
                accuracy=0

                for ii, (inputs_validation,labels_validation) in enumerate(dataloaders[1]):
                       #Zero parameter gradients
                       optimizer.zero_grad()
                       #Select CUDA processing if it's suppoerted by the environment 
                       if torch.cuda.is_available():
                            inputs_validation, labels_validation = inputs_validation.to('cuda') , labels_validation.to('cuda') 
                            model.to('cuda:0') 
                       #Use input    
                       else:
                            pass 
                       #Gradients are turned off
                       with torch.no_grad():    
                            outputs = model.forward(inputs_validation)
                            validation_loss = criterion(outputs,labels_validation)
                            ps = torch.exp(outputs).data
                            equality = (labels_validation.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                validation_loss = validation_loss / len(dataloaders[1])
                accuracy = accuracy /len(dataloaders[1])

                print("Training_Loss: {:.4f}".format(running_loss/print_every),
                      "Validation_Loss {:.4f}".format(validation_loss),
                      "Accuracy: {:.4f}".format(accuracy),
                     )

                running_loss = 0
                #Turning training back 
                model.train()
                
#All the above functions are called and executed are in main()              
def main():
    print('Model training started') 
    
    #Record the starting time
    starting_time = time.time()
    #Get keyword Args for training
    args = parse_args()
    #Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #Train
    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),                                                               transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) ])
    #Validation
    validataion_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),                                                                                transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) ]) 
    #Test
    testing_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),                                                                                transforms.RandomHorizontalFlip(),transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) ])  
   
    #Loaders training
    train_loaders =  [ImageFolder(train_dir, transform=training_transforms),
                      ImageFolder(val_dir, transform=validataion_transforms),
                      ImageFolder(test_dir, transform=testing_transforms)]
    #Data Loaders
    dataloaders = [torch.utils.data.DataLoader(train_loaders[0], shuffle=True, batch_size=50),
                   torch.utils.data.DataLoader(train_loaders[1], shuffle=True, batch_size=50),
                   torch.utils.data.DataLoader(train_loaders[2], shuffle=True, batch_size=50)]
   
    #Load pretrained model
    model = getattr(models, args.arch)(pretrained=True)
     
    #Classifier building    
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg11":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.6)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    
    elif args.arch == "alexnet":
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('drop', nn.Dropout(p=0.5)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    

    model.classifier = classifier
    #Defining loss and optimizer 
    criterion = nn.NLLLoss() 
    #Using Adam optimizer to avoid local minima 
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = train_loaders[0].class_to_idx
    #Checking for GPU
    gpu = args.gpu 
    #Train the classifier layers using backpropogation
    print('\Model training')
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    #Saving the model in new location
    path = args.save_dir
    save_checkpoint(path, model, optimizer, args, classifier)
    print('Checkpoint saved')
    
    print('The model has been trained successfully!')

#Running the program
if __name__ == "__main__":
    main()

