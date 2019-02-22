import torch
from torch import nn
from torch import optim
from torchvision import models
from collections import OrderedDict
from PIL import Image
import numpy as np
from workspace_utils import active_session
import utilities


def build_base_model(arch):
    if arch == 'vgg11':
        basemodel = models.vgg11(pretrained=True)
    elif arch == 'vgg13':
        basemodel = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        basemodel = models.vgg16(pretrained=True)
    elif arch == 'vgg19':
        basemodel = models.vgg19(pretrained=True)
    elif arch == 'vgg11_bn':
        basemodel = models.vgg11_bn(pretrained=True)
    elif arch == 'vgg13_bn':
        basemodel = models.vgg13_bn(pretrained=True)
    elif arch == 'vgg16_bn':
        basemodel = models.vgg16_bn(pretrained=True)
    elif arch == 'vgg19_bn':
        basemodel = models.vgg19_bn(pretrained=True)
    else:
        print("Base model invalid, exiting")
        exit(1)
    for param in basemodel.parameters():
        param.requires_grad = False
    return basemodel


def build_classifier(units):
    # Build final stage with variable number of hidden layers provided in units list
    # Input layer to classifier
    architecture = OrderedDict([('fc1', nn.Linear(units[0], units[1])),
                                ('relu1', nn.ReLU()),
                                ('dropout1', nn.Dropout(0.5))])
    # Hidden layers
    for layer in range(len(units) - 3):
        architecture['fc{}'.format(layer+2)] = nn.Linear(units[layer+1], units[layer+2])
        architecture['relu{}'.format(layer+2)] = nn.ReLU()
        architecture['dropout{}'.format(layer+2)] = nn.Dropout(0.5)
    # Output layer
    architecture['fc{}'.format(len(units) - 1)] = nn.Linear(units[-2], units[-1])
    architecture['output'] = nn.LogSoftmax(dim=1)
    classifier = nn.Sequential(architecture)
    return classifier


def train(model, learning_rate, epochs, gpu, trainloader, validationloader):
    device = torch.device("cuda:0" if gpu else "cpu")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)
    running_loss = 0
    steps = 0
    print_every = 20

    with active_session():
        for e in range(epochs):
            for images, labels in trainloader:
                steps += 1
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loss = 0
                    validation_accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for images, labels in validationloader:
                            images, labels = images.to(device), labels.to(device)
                            log_ps = model(images)
                            validation_loss += criterion(log_ps, labels)
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            validation_accuracy += torch.mean(equals.type(torch.FloatTensor))

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validationloader)),
                          "Validation Accuracy: {:.3f}".format(validation_accuracy/len(validationloader)))
                    running_loss = 0
                    model.train()
    return model


def test(model, testloader, gpu):
    device = torch.device("cuda:0" if gpu else "cpu")
    criterion = nn.NLLLoss()
    test_loss = 0
    test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model(images)
            test_loss += criterion(log_ps, labels)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor))

    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(test_accuracy/len(testloader)))


def save_checkpoint(model, basemodel, save_dir, arch, classifier_units, learning_rate, epochs, train_data):
    checkpoint = {'base_model': basemodel,
                  'classifier': {'input_units': classifier_units[0],
                                 'output': classifier_units[-1],
                                 'hidden_layers': classifier_units[1:-1]
                                },
                  'hyperparameters': {
                      'learning_rate': learning_rate,
                      'epochs': epochs},
                  'state_dict': model.state_dict(),
                  'class_to_idx': train_data.class_to_idx
                 }
    save_path = '{0}/{1}.pth'.format(save_dir, arch)
    torch.save(checkpoint, save_path)
    return save_path
    

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['base_model']
    for param in model.parameters():
        param.requires_grad = False
    # Build a list of the number of units in each layer of the classifier
    units = [checkpoint['classifier']['input_units']] \
            + checkpoint['classifier']['hidden_layers'] \
            + [checkpoint['classifier']['output']]
    classifier = build_classifier(units)
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    learning_rate = checkpoint['hyperparameters']['learning_rate']
    epochs = checkpoint['hyperparameters']['epochs']
    return model


def predict(image_path, model, topk, gpu):
    # Predict the class (or classes) of an image using a trained deep learning model.
    device = torch.device("cuda:0" if gpu else "cpu")
    model.to(device)
    image = Image.open(image_path)
    processed_image = utilities.process_image(image)
    processed_image_tensor = torch.from_numpy(processed_image).view(1, 3, 224, 224).float().to(device)
    model.eval()
    with torch.no_grad():
        log_ps = model(processed_image_tensor)
    ps = torch.exp(log_ps)
    top_p, top_idx = ps.topk(topk, dim=1)
    top_p = np.array(top_p)
    top_p = np.reshape(top_p, [topk, ]).tolist()
    top_idx = np.array(top_idx)
    top_idx = np.reshape(top_idx, [topk,]).tolist()
    idx_to_class = dict([[v,k] for k,v in model.class_to_idx.items()])
    top_classes = list([idx_to_class[k] for k in top_idx])
    return top_p, top_classes
