import pickle
import numpy as np
import os

input_folder = '/path-to/adversarial_continual/data/miniimagenet/'
output = '/path-to/adversarial_continual/data/miniimagenet/'

# download pkls from https://github.com/renmengye/few-shot-ssl-public
trainset = pickle.load( open( os.path.join(input_folder, "mini-imagenet-cache-train.pkl"), "rb" ) )
valset = pickle.load( open( os.path.join(input_folder, "mini-imagenet-cache-val.pkl"), "rb" ) )
testset = pickle.load( open( os.path.join(input_folder, "mini-imagenet-cache-test.pkl"), "rb" ) )

def gothrough(oneset, start_index=0):
	images, labels = [], []
	for i, label in enumerate(list(oneset['class_dict'].keys())):
		i += start_index
		print('task {} has {} samples'.format(i, len(oneset['class_dict'][label])))
		for x in oneset['class_dict'][label]:
			images.append(oneset['image_data'][x,:,:,:])
			labels.append(i)
	return images, labels

images, labels = [], []
subimages, sublabels = gothrough(trainset)
images += subimages
labels += sublabels
start_index = len(list(trainset['class_dict'].keys()))
subimages, sublabels = gothrough(valset, start_index)
images += subimages
labels += sublabels
start_index = len(list(trainset['class_dict'].keys()))+len(list(valset['class_dict'].keys()))
subimages, sublabels = gothrough(testset, start_index)
images += subimages
labels += sublabels

data_dict = {'images': images, 'labels': labels}

with open(os.path.join(output, 'data.pkl'), 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)