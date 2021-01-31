import os
import string

DEBUG = True
# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# extract filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	puc_table = str.maketrans('', '', string.punctuation)
	# prepare translation table for removing english letter
	en_table = str.maketrans('', '', string.ascii_letters)
	table = {**puc_table, **en_table}
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			desc = desc.replace('।', '')
			# tokenize
			desc = desc.split()
			# remove punctuation and english letter from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isdigit() is False]
			# store as string
			desc_list[i] =  ' '.join(desc)

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def process_dataset(config):
	caption_file_path = config.token_path
	clean_caption_save_path = config.descriptions_file_path
	doc = load_doc(caption_file_path)
	if DEBUG:
		print('Raw descriptions : ', doc[:300])
	# parse descriptions
	descriptions = load_descriptions(doc)
	print('Total Loaded Captions: %d ' % len(descriptions))
	if DEBUG:
		print('Key of some captions : ', list(descriptions.keys())[:5])
		key = '1002674143_1b742ab4b8'
		print('Captions for this key {}:\n{}'.format(key, descriptions[key]))
	# clean descriptions
	clean_descriptions(descriptions)
	if DEBUG:
		print('Captions after cleaning : ', descriptions[key])
	# save captions after cleaning
	save_descriptions(descriptions, clean_caption_save_path)
	print("Clean captions saved to disk : Done")


if __name__ == '__main__':
	from config import Config
	config = Config()
	process_dataset(config)