import pickle
# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load a pre-defined list of photo identifiers
def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(des_file_path, image_list_path):
    # load document
    doc = load_doc(des_file_path)
    image_name_list = load_set(image_list_path)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in image_name_list:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions

def get_captions(descriptions):
        # Create a list of all the training captions
    all_captions = []
    for key, val in descriptions.items():
        for cap in val:
            all_captions.append(cap)
    return all_captions

def get_vocab(descriptions, word_count_threshold=10):
    # Consider only words which occur at least 10 times in the corpus
    all_captions = get_captions(descriptions)
    word_counts = {}
    nsents = 0
    for sent in all_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
    return vocab

def convert_ixtoword_and_wordtoix(vocab):
    ixtoword = {}
    wordtoix = {}

    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1
    return ixtoword, wordtoix
# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# calculate the length of the description with the most words
def get_max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

if __name__ == '__main__':
    from config import Config
    config = Config()
    train_descriptions = load_clean_descriptions(config.descriptions_file_path, config.train_file_path)
    print("Total discriptions : ", len(train_descriptions))
    vocab = get_vocab(train_descriptions)
    print("Total vocab : ", len(vocab))
    ixtoword, wordtoix = convert_ixtoword_and_wordtoix(vocab)
    vocab_size = len(ixtoword)
    print("Vocab size : ", vocab_size)
    max_length = get_max_length(train_descriptions)
    print("Max caption length : ", max_length)