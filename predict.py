import argparse
import json
import modellib
import utilities


if __name__ == '__main__':
    # Parse arguments and initialise
    parser = argparse.ArgumentParser(description='Predict type of flower from a picture.')
    parser.add_argument('imagepath', type=str,
                        help='Path to the image file')
    parser.add_argument('checkpoint_file', type=str,
                        help="Path to the model's trained checkpoint file")
    parser.add_argument('--top_k', type=int, dest='top_k', default=1,
                        help="The number of most likely matches to return")
    parser.add_argument('--category_names', type=str, dest='category_names', default='cat_to_name.json',
                        help="Path to file listing known flower types")
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true',
                        help="Switch to enable GPU for inference (if available)")
    args = parser.parse_args()
    print("Initialising prediction with the following arguments:")
    print("Image to examine is: {}".format(args.imagepath))
    print("Prediction model is: {}".format(args.checkpoint_file))
    print("The model will return the {} top possible matches and probabilities".format(args.top_k))
    print("The list of possible categories used is: {}".format(args.category_names))
    print("{} that GPU use will be requested for prediction".format(args.gpu))

    # Check category names file exists
    try:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    except IOError:
        print("File {} not found, reverting to cat_to_name.json".format(args.category_names))
        cat_to_name = json.load('cat_to_name.json')
        args.category_names = 'cat_to_name.json'
    args.gpu = utilities.choose_device(args.gpu)
    model = modellib.load_checkpoint(args.checkpoint_file)
    probs, classes = modellib.predict(args.imagepath, model, args.top_k, args.gpu)

    print("The picture is predicted to be the following, with the accompanying probabilities:")
    for i in range(len(classes)):
        print("Predicted to be a {0}, with probability {1}".format(
                                                            cat_to_name[classes[i]],
                                                            probs[i]))
