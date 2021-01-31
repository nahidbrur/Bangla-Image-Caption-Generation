import os
import glob
from time import time
import numpy as np
from PIL import Image
from pickle import dump, load
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import image

def load_dataset(filename, image_dir):
	# Read the train image names in a set
	image_names = set(open(filename, 'r').read().strip().split('\n'))
	# all image names in the image directory
	image_paths = glob.glob(image_dir + '/*.jpg')
	# Create a list of all the training images with their full path names
	images = []
	for image_path in image_paths: # img is list of full path names of all images
	    if image_path.split('/')[-1] in image_names: # Check if the image belongs to training set
	        images.append(image_path.split('/')[-1]) # Add it to the list of train images
	return images

def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

# Function to encode a given image into a vector of size (2048, )
def encode(model, image):
    image = preprocess(image) # preprocess the image
    fea_vec = model.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def encode_images_into_pickle(config):
	train_images_file = config.train_file_path
	test_images_file = config.test_file_path
	image_dir = config.image_dir

	train_images = load_dataset(train_images_file, image_dir)
	test_images = load_dataset(test_images_file, image_dir)

	print("Total train images : ", len(train_images))
	print("Total test images : ", len(test_images))

	# Call the funtion to encode all the train images
	# This will take a while on CPU - Execute this only once
	start = time()
	encoding_train = {}
	print("Loading InceptionV3 Model...")
	# Load the inception v3 model
	model = InceptionV3(weights='imagenet')
	# Create a new model, by removing the last layer (output layer) from the inception v3
	model = Model(model.input, model.layers[-2].output)
	print("Train Imge encoding start..")
	for train_image in train_images:
	    encoding_train[train_image] = encode(model, os.path.join(image_dir, train_image))
	# Save the bottleneck train features to disk
	print("Saving encoded...")
	with open(config.train_features_file_path, "wb") as encoded_pickle:
	    dump(encoding_train, encoded_pickle)
	print("Finish encoding. Total time taken (s) = ", time()-start)

	start = time()
	encoding_test = {}
	print("Test Imge encoding start..")
	for test_image in test_images:
	    encoding_test[test_image] = encode(model, os.path.join(image_dir, test_image))
	# Save the bottleneck train features to disk
	print("Saving encoded...")
	with open(config.test_features_file_path, "wb") as encoded_pickle:
	    dump(encoding_test, encoded_pickle)
	print("Finish encoding. Total time taken (s) = ", time()-start)

if __name__ == '__main__':
	from config import Config
	config = Config()
	# encode train and test images and save to disk
	encode_images_into_pickle(config)