"""
helper.py
Helper functions for Image Classifier Project notebook

Includes: 

class Network(nn.Module)
validation(model, dataloader, criterion)
"""

import torch
import time
from torch import nn
import torch.nn.functional as F
from torchvision import models

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_p = 0.5):
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
    
def build_model(pre_trained_network=models.vgg16(pretrained=True),
               input_size=25088, output_size=102, hidden_layers=[4096, 512]):

    # Use GPU if it's available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(42)
    else:
        device = torch.device("cpu")
        
    # Load a pre-trained network
    model = pre_trained_network
    
    # Freeze features parameters so backpropagation isn't performed
    for param in model.parameters():
        param.require_grad = False
        
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    model.classifier = Network(input_size, output_size, hidden_layers)

    model.to(device)
    
    return model, device

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
    
def train(model, device, trainloader, validloader, criterion, optimizer, epochs=1, print_every=5, max_steps=None):
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

            if steps % print_every == 0:
                model.eval()
                
                valid_loss, valid_accuracy = validation(model, device, validloader, criterion)
                train_loss = running_loss/print_every
                train_accuracy = running_accuracy/print_every
                
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
        
    results_dict = {'train_losses':train_losses, 'train_accuracies':train_accuracies, 
                    'valid_losses':valid_losses, 'valid_accuracies':valid_accuracies,
                    'epoch_durations':epoch_durations}
    
    return results_dict

def save_checkpoint(filepath, model, results_dict):
    input_size = model.classifier.hidden_layers[0].in_features
    output_size = model.classifier.output.out_features
    hidden_layers = []
    for layer in model.classifier.hidden_layers:
        hidden_layers.append(layer.out_features)
        
    checkpoint = {'input_size':input_size,
                 'output_size':output_size,
                 'hidden_layers':hidden_layers,
                 'state_dict':model.state_dict(),
                 'results_dict':results_dict}
    torch.save(checkpoint, filepath)
    
def load_checkpoint(filepath, pre_trained_network=models.vgg16(pretrained=True)):
    checkpoint = torch.load(filepath)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
        
    model, device = build_model(pre_trained_network=pre_trained_network,
                                input_size=input_size, output_size=output_size, 
                                hidden_layers=hidden_layers)

    model.load_state_dict(checkpoint['state_dict'])

    previous_results = checkpoint['results_dict']
    
    return model, device, previous_results