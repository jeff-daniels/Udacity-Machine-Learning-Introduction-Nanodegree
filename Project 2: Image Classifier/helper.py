"""
helper.py
Helper functions for Image Classifier Project notebook

"""

import torch
import time
from torch import nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import json
import numpy as np

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_p):
        '''
        Builds a feedforward network with hidden layers
        
        Arguments
        -----------
        input_size: integer, size of the input layer
        output_size: integer, size of the output layer
        hidden_layers: list of integers, sizes of the hidden layers
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a varying number of additional hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p= dropout_p)
        
    def forward(self, x):
        ''' Forward pass through the networdk, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim = 1)
    
def build_model(architecture='vgg16', gpu = False,
               input_size=25088, output_size=102, hidden_layers=[4096, 512], 
               dropout_p = 0.5):
    
    # Define network architecture
    if architecture == 'vgg16':
        pretrained_network = models.vgg16(pretrained=True)
    elif architecture == 'alexnet':
        pretrained_network = models.alexnet(pretrained=True)
    elif architecture == 'resnet18':
        pretrained_network = models.resnet18(pretrained=True)
    else:
        print('Unknown Model')
        pretrained_network = None
    
    # Use GPU if it's available
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device("cpu")
        
    # Load a pre-trained network
    model = pretrained_network
    
    # Freeze features parameters so backpropagation isn't performed
    for param in model.parameters():
        param.require_grad = False
        
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    model.classifier = Network(input_size, output_size, hidden_layers, dropout_p)

    model.to(device)
    
    return model, device, architecture

def validation(model, device, dataloader, criterion):
    loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    loss = loss/len(dataloader)
    accuracy = accuracy/len(dataloader)
    return loss, accuracy
    
def train(model, device, trainloader, validloader, criterion, optimizer, 
          epochs=1, print_every=5, max_steps=None, performance_dict = {}):
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    steps = 0
    running_loss = 0
    running_accuracy = 0
    train_losses, train_accuracies, valid_losses, valid_accuracies, epoch_durations = [], [], [], [], []
    
    for epoch in range(epochs):
        start = time.time()
        model.train()
        if steps == max_steps:
            break
        for inputs, labels in trainloader:
            steps += 1
            # Stop early for testing
            if steps == max_steps:
                break
                
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            # Print out in-training performance
            if steps % print_every == 0:
                model.eval()
                
                # Calculate performance
                valid_loss, valid_accuracy = validation(model, device, validloader, criterion)
                train_loss = running_loss/print_every
                train_accuracy = running_accuracy/print_every
                
                # Update performance lists
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                valid_losses.append(valid_loss)
                valid_accuracies.append(valid_accuracy)
                
                print(f"Step {steps}.."
                      f"Epoch {epoch+1}/{epochs}.."
                      f"Train loss: {train_loss/print_every:.3f}.."
                      f"Train Accuracy: {train_accuracy:.3f}.."
                      f"Validation loss: {valid_loss:.3f}.."
                      f"Validation Accuracy: {valid_accuracy:.3f}"
                     )
                
                running_loss = 0
                running_accuracy =0
                model.train()
                
        end = time.time()
        epoch_durations.append(end-start)
        print(f"Epoch duration: {end-start}")
    
    # Create or update performance_dict
    keys = ['train_losses', 'train_accuracies', 'valid_losses', 'valid_accuracies', 'epoch_durations']
    lsts = [train_losses, train_accuracies, valid_losses, valid_accuracies, epoch_durations]
    for key, lst in zip(keys, lsts):
        if key in performance_dict.keys():
            for item in lst:
                performance_dict[key].append(item)
        else:
            performance_dict[key] = lst
    
    num_epochs = len(performance_dict['epoch_durations'])
    performance_dict['num_epochs'] = num_epochs
    
    # Create training_dict
    input_size = model.classifier.hidden_layers[0].in_features
    output_size = model.classifier.output.out_features
    hidden_layers = []
    for layer in model.classifier.hidden_layers:
        hidden_layers.append(layer.out_features)

    dropout_p = model.classifier.dropout.p
    
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    
    training_dict = {'input_size':input_size, 'output_size':output_size, 
                     'hidden_layers':hidden_layers,
                     'criterion':criterion, 'optimizer':optimizer, 
                     'lr':lr, 'dropout_p':dropout_p}

    return performance_dict, training_dict

def plot_results(results):
    import matplotlib.pyplot as plt
    % matplotlib inline
    plt.plot(results['valid_losses'], 'b')
    plt.plot(results['train_losses'], 'r')

def compare_hyperparameters(filepaths):
    perf = {}
    perf['Epochs'] = [5, 5, 5, 5]
    perf['Hidden Layers'] = ['4096x512', '4096x512', '4096x512', '512x256']
    perf['Dropout Proportion'] = [0.5, 0.5, 0.5, 0.5]
    perf['Learning Rate'] = [0.001, 0.003, 0.0003, 0.0003]
    
    min_valid_loss = []
    max_valid_accuracy = []
    
    for fp in filepaths:
        with open(fp, 'r') as f:
            perf_dict = json.load(f)
            min_valid_loss.append(min(perf_dict['valid_losses']))
            max_valid_accuracy.append(max(perf_dict['valid_accuracies']))
    
    perf['Minimum Validation Loss'] = min_valid_loss
    perf['Maximum Validation Accuracy'] = max_valid_accuracy
     
    df = pd.DataFrame(perf)
    return df    

def save_checkpoint(chkpt_filepath, perf_filepath, 
                    model, architecture, performance_dict, training_dict, mapping_dict):
    
    # No need to save as gpu
    model.to('cpu')
    
    # Save the checkpoint
    checkpoint = {'state_dict':model.state_dict(),
                  'architecture':architecture,
                  'performance_dict':performance_dict, 
                  'training_dict':training_dict, 
                  'mapping_dict':mapping_dict}
    torch.save(checkpoint, filepath)
    
    # Save the performance dict in an additional file for reference
    json = json.dumps(performance_dict)
    f = open(perf_filepath,"w")
    f.write(json)
    f.close()

    return None
    
def load_checkpoint(filepath, gpu=False):
    checkpoint = torch.load(filepath)
    architecture = checkpoint['architecture']
    input_size = checkpoint['training_dict']['input_size']
    output_size = checkpoint['training_dict']['output_size']
    hidden_layers = checkpoint['training_dict']['hidden_layers']
    dropout_p = checkpoint['training_dict']['dropout_p']
        
    model, device, architecture = build_model(architecture=architecture, gpu=gpu,
                                input_size=input_size, output_size=output_size, 
                                hidden_layers=hidden_layers,
                                dropout_p=dropout_p)

    model.load_state_dict(checkpoint['state_dict'])

    performance_dict = checkpoint['performance_dict']
    mapping_dict = checkpoint['mapping_dict']
    model.class_to_idx = mapping_dict
    
    return model, device, architecture,  performance_dict, mapping_dict

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size_short_side = 256
    width = 224
    height = 224
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Resize Image
    ratio = float(size_short_side/image.size[0])
    new_size = (size_short_side, int(image.size[1]*ratio))
    image = image.resize(new_size)
    
    # crop out center 224x224 portion
    left_offset = int((image.size[0]-width)/2)
    upper_offset = int((image.size[1]-height)/2)
    box = (left_offset, upper_offset, left_offset+width, upper_offset+height)
    image = image.crop(box)

    # Convert to numpy array, normalize, and reorder dimensions  
    np_image = np.array(image)
    np_image = (np_image/255 - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, label_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        return probabilities and classes
    '''
    # Implement the code to predict the class from an image file
    
    # Load image
    image = Image.open(image_path)
    
    # Load label mapping
    with open(label_path, 'r') as f:
        cat_to_name = json.load(f)

    # Convert image into a pytorch tensor
    np_image = process_image(image)
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    tensor_image = tensor_image.unsqueeze(dim=0) # torch.Tensor, torch.Size([3, 224, 224])

    # Make a prediction
    tensor_image = tensor_image.to(device)
    with torch.no_grad():
        model.eval()
        output = model.forward(tensor_image)

    # Calculate probabilites and most likely classes
    probabilities = torch.exp(output)
    top_p, top_class = probabilities.topk(topk, dim=1)

    # Convert top_p to a list
    probs = list(top_p.cpu().numpy().squeeze())

    # Figure out the flower names and classes
    flower_classes, flower_names = [], []
    for idx in top_class.cpu().numpy().squeeze():
        flower_class = list(model.class_to_idx.keys())[list(model.class_to_idx.values()).index(idx)]
        flower_name = cat_to_name[flower_class]
        flower_classes.append(flower_class)
        flower_names.append(flower_name)

    classes = flower_classes
        
    return probs, classes, flower_names

def display_and_plot_classes(image_path, classes):
    original_image = Image.open(image_path)
    np_image = process_image(original_image)

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np_image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    # Figure out the flower names
    flower_names = []

    for flower_class in classes:
        flower_name = cat_to_name[flower_class]
        flower_names.append(flower_name)

    # Plot the original image and the probabilites of the top classes
    fig, (ax1, ax2) = plt.subplots(figsize = (15,4), ncols=2)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Actual Flower Name Goes Here')
    ax2.barh(np.arange(5), top_p.cpu().numpy().squeeze())
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(flower_names)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)