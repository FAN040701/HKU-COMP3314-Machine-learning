from __future__ import print_function, division

import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import numpy as np

import torch.nn as nn

# TODO: You can modify the network architecture
class Net(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(256),

            nn.Flatten(), 
            nn.Linear(256*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
            )
        
    def forward(self, xb):
        return self.network(xb)
    
    
# TODO: You can try different augmentation strategies
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0,0.1), scale=(0.9,1.1), shear=10),
        transforms.ColorJitter(brightness=0.5, contrast=(1.1,1.2), saturation=0.4, hue=0.1),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

    ]),
    'test': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}
    
data_dir = 'data' # Suppose the dataset is stored under this folder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'test']} # Read train and test sets, respectively.

train_dataloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=128,
                                             shuffle=True, num_workers=4)

test_dataloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=128,
                                             shuffle=False, num_workers=4)

train_size =len(image_datasets['train'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Set device to "cpu" if you have no gpu

def train_test(model, criterion, optimizer, scheduler, num_epochs=25):
    train_loss = []
    train_accuracy = []
    val_loss = [] 
    val_accuracy = []
    history = dict()
    model.train()
    for epoch in range(num_epochs):
        running_training_loss = 0.0
        running_training_accuracy = 0.0
        iteration_training_loss = 0.0
        total_training_predictions = 0
       
        start_time = time.time()
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_training_loss += loss.item()*inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_training_predictions += labels.size(0)
            running_training_accuracy += (predicted == labels).sum().item()
            iteration_training_loss += loss.item() 
            if (i+1) % 100 == 0:
                print('Epoch:[%d]-Iteration:[%d], training loss: %.3f' %
                      (epoch + 1,i+1,iteration_training_loss/(i+1)))
        end_time = time.time()
        print('Time cost of one epoch: [%d]s' % (end_time-start_time))
        
        epoch_training_accuracy = running_training_accuracy / train_size*100
        epoch_training_loss = running_training_loss / train_size
        
        print('Epoch:[%d], training accuracy: %.1f, training loss: %.3f' %
              (epoch + 1,epoch_training_accuracy, epoch_training_loss))
        
        train_loss.append(epoch_training_loss)
        train_accuracy.append(epoch_training_accuracy)
        
        if epoch_training_accuracy >= 90:
            optimizer = optim.AdamW(model_ft.parameters(), lr=1e-4) 

        scheduler.step()
        
    print('Finished Training')

    history['train_loss'] = train_loss
    history['train_accuracy'] = train_accuracy

    correct = 0
    total = 0
    model.eval()
    # for per class accuracy
    y_pred = []
    y_true = []
    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #caculate the accuracy for each class
            y_pred.extend(predicted.view(-1).tolist())
            y_true.extend(labels.view(-1).tolist())
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on test images: {accuracy:.2f}%')
    
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Calculate the accuracy for each class
    class_accuracy = 100 * conf_matrix.diagonal() / conf_matrix.sum(1)
    # Print the accuracy for each class
    for idx, acc in enumerate(class_accuracy):
        print(f'Accuracy for class {idx}: {acc:.2f}%')
    
    return history, accuracy

'''
def calculate_per_class_accuracy(model, test_dataloader, device):
    model.eval()  # Set the model to evaluation mode
    y_pred = []
    y_true = []

    # No gradient is needed for evaluation
    with torch.no_grad():
        for inputs, classes in test_dataloader:
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.view(-1).tolist())
            y_true.extend(classes.view(-1).tolist())

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Calculate the accuracy for each class
    class_accuracy = 100 * conf_matrix.diagonal() / conf_matrix.sum(1)
    return class_accuracy'''

if __name__ == '__main__':
    end = time.time()
    model_ft = Net().to(device) # Model initialization
    print(model_ft.network)
    criterion = nn.CrossEntropyLoss() # Loss function initialization

    # TODO: Adjust the following hyper-parameters: initial learning rate, decay strategy of the learning rate, 
    #       number of training epochs
    optimizer_ft = optim.AdamW(model_ft.parameters(), lr=1e-3) # The initial learning rate is 1e-3

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.6) 
    
    history, accuracy = train_test(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
               num_epochs=25)
    
    print("time required %.2fs" %(time.time() - end))
    