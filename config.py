import os

class Config():

	root_dataset_dir = 'dataset'

	image_dir = root_dataset_dir + '/Flicker8k_Dataset'
	train_file_path = root_dataset_dir + '/Flickr8k_text/Flickr_8k.trainImages.txt'
	test_file_path =root_dataset_dir + '/Flickr8k_text/Flickr_8k.testImages.txt'
	token_path = root_dataset_dir + '/Flickr8k_text/Flickr8k.bn_token.txt'

	train_features_file_path = root_dataset_dir + '/train_features.pickle'
	test_features_file_path = root_dataset_dir + '/test_features.pickle'

	descriptions_file_path = root_dataset_dir + '/bn_descriptions.txt'
	word2vec_file_path = root_dataset_dir + '/word2vec_bangla.txt'

	checkpoint_dir = root_dataset_dir + '/models'

	embedding_dim = 300
	batch_size = 10
	epochs = 100
