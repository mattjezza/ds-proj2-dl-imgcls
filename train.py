import argparse
import modellib
import utilities


if __name__ == '__main__':
    # Parse arguments and initialise
    parser = argparse.ArgumentParser(description='Train a machine learning model on a data set and save the model.')
    parser.add_argument('data_dir', type=str,
                        help='Path to the directory containing training data')
    parser.add_argument('--save_dir', type=str, dest='save_dir', default='.',
                        help="Path to directory  where the model's checkpoint file will be saved")
    parser.add_argument('--arch', type=str, dest='arch', default='vgg11',
                        help="The base architecture for the model. "
                             "Can be any of the following: "
                             "vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn")
    parser.add_argument('--hidden_units', dest='hidden_units', default=[], action='append', type=int,
                        help="Number of units in a hidden layer in the final stage classifier. "
                             "Specify this option once for every hidden layer required.")
    parser.add_argument('--learning_rate', type=float, dest='learning_rate', default='0.001',
                        help="Learning rate to be used during training")
    parser.add_argument('--epochs', type=int, dest='epochs', default=5,
                        help="Number of epochs for training")
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true',
                        help="Switch to enable GPU for training (if available)")
    args = parser.parse_args()
    print("Creating and training new model with the following arguments:")
    print("Data directory is: {}".format(args.data_dir))
    print("Checkpoint save directory is: {}".format(args.save_dir))
    print("Base architecture for the model: {}".format(args.arch))
    if args.arch not in ['vgg11', 'vgg13', 'vgg16', 'vgg19',
                         'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']:
        print("Base architecture {} is not a valid choice. Please choose one of:"
              "vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn". format(args.arch))
        exit(1)
    if len(args.hidden_units) == 0:
        print("No hidden layers specified in classifier, selecting default (1024)")
        args.hidden_units.append(1024)
    print("The final stage classifier in the model "
          "will have the following hidden layer structure: {}".format(args.hidden_units))
    print("Learning rate: {}".format(args.learning_rate))
    print("Epochs: {}".format(args.epochs))
    print("{} that GPU use will be requested for prediction".format(args.gpu))

    args.gpu = utilities.choose_device(args.gpu)
    trainloader, testloader, validationloader, train_data = utilities.set_training_data(args.data_dir)
    basemodel = modellib.build_base_model(args.arch)
    classifier_units = args.hidden_units
    input_units = basemodel.state_dict()['classifier.0.weight'].size()[1]
    classifier_units.insert(0, input_units)
    # Number of output units is a fixed number (102).
    # A good improvement would be to pass a category->name file (like cat_to_name.json)
    # into the trainer script as an argument (as in the predictor script).
    # Then the number of output units could be calculated directly from this rather than assuming 102.
    output_units = 102
    classifier_units.append(output_units)
    model = basemodel
    model.classifier = modellib.build_classifier(classifier_units)
    model = modellib.train(model, args.learning_rate, args.epochs, args.gpu, trainloader, validationloader)
    print("Model training complete.")
    modellib.test(model, testloader, args.gpu)
    print("Model testing complete.")
    saved_model = modellib.save_checkpoint(model, basemodel, args.save_dir, args.arch, classifier_units,
                                           args.learning_rate, args.epochs, train_data)
    print("Model saved as {}".format(saved_model))
