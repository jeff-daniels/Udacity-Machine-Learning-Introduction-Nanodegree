import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import cl_helper
from workspace_utils import active_session, keep_awake

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="data directory of the training files", type=str)
parser.add_argument("--save_dir", default='', help="directory to save the training checkpoint", type=str)
parser.add_argument("--arch", default = 'vgg16', 
                    help="model archictecture, choose 'vgg16', 'alexnet', 'resnet18'", type=str)
parser.add_argument("--learning_rate", default=0.0003, help="learning rate", type=float)
parser.add_argument("--hidden_units", default=[4096, 512], help="hidden units in list form", type=list)
parser.add_argument("--dropout_p", default=0.5, help="dropout rate", type=float)
parser.add_argument("--epochs", default=1, help="number of epochs to train", type=int)
parser.add_argument("--max_steps", default=None, 
                    help="maximum of batches to train, None=Unlimited", type=int)
parser.add_argument("--print_every", default=20, 
                    help="number of batches between validations and print updates", type=int)
parser.add_argument("--gpu", help="enable gpu for training", action="store_true")
args = parser.parse_args()

# Define your transforms for the training, validation, and testing sets
batch_size = 64
mean_norm = [0.485, 0.456, 0.406]
std_norm = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = mean_norm,
                                                          std = std_norm)])

test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = mean_norm,
                                                            std = std_norm)])

# Load the datasets with ImageFolder
train_dir = args.data_dir + '/train'
valid_dir = args.data_dir + '/valid'

train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = test_valid_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

# Construct a model and its training parameters
model, device, architecture = cl_helper.build_model(architecture=args.arch, gpu=args.gpu,
                                      hidden_layers=args.hidden_units, dropout_p=args.dropout_p)
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)

# Train your network

with active_session():
    print('Training in Progress.....')
    print('GPU status: {}'.format(args.gpu))
    performance_dict, training_dict  = cl_helper.train(model, device, trainloader, validloader, 
                                                    criterion, optimizer, 
                                                    epochs = args.epochs, 
                                                    print_every = args.print_every, 
                                                    max_steps=args.max_steps)

print('Training Complete')

# TODO: Save the checkpoint 
filepath = args.save_dir + 'checkpoint.pth'

model.class_to_idx = train_data.class_to_idx
mapping_dict = model.class_to_idx
cl_helper.save_checkpoint(filepath, model, architecture, performance_dict, training_dict, mapping_dict)
print('Check point saved in {}'.format(filepath))