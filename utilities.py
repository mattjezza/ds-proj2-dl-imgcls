import torch
from torchvision import datasets, transforms
import numpy as np


def set_training_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.RandomChoice([transforms.RandomResizedCrop(224),
                                                                  transforms.CenterCrop(224)]),
                                          transforms.RandomApply([transforms.RandomRotation(90),
                                                                  transforms.ColorJitter(),
                                                                  transforms.RandomHorizontalFlip(),
                                                                  transforms.RandomVerticalFlip(),
                                                                  transforms.RandomGrayscale()]),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_and_vldtn_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_and_vldtn_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=test_and_vldtn_transforms)

    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    
    return trainloader, testloader, validationloader, train_data


def process_image(image):
    # Scales, crops, and normalizes a PIL image for a PyTorch model,
    # returns a Numpy array

    image_size = image.size
    if image_size[0] > image_size[1]:
        scale_factor = 256 / image_size[1]
    else:
        scale_factor = 256 / image_size[0]
    newsize = (int(image_size[0]*scale_factor), int(image_size[1]*scale_factor))
    resized_image = image.resize(newsize)
    leftcropborder = (resized_image.size[0]-224) // 2
    topcropborder = (resized_image.size[1]-224) // 2
    lowercropborder = topcropborder + 224
    rightcropborder = leftcropborder + 224
    cropwindow = (leftcropborder, topcropborder, rightcropborder, lowercropborder)
    cropped_image = resized_image.crop(cropwindow)
    np_image = np.array(cropped_image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image


def choose_device(gpu):
    # Choose GPU or CPU
    if gpu is True:
        if torch.cuda.is_available():
            print("GPU requested and will be used for prediction")
            device = True
        else:
            print("GPU requested but no GPU found. Using CPU for prediction")
            device = False
    else:
        print("GPU use not requested, CPU will be used for prediction")
        device = False
    return device
