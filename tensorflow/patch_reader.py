import collections
import os
import scipy.io as sio
import tensorflow as tf
import numpy as np
import numpy.matlib as npm
import random
import cv2
from tqdm import tqdm
from preprocess import normalize_data

from scipy.spatial import distance

class SiameseDataSet(object):
    def __init__(self,
                 base_dir,
                 test = False
                 ):

        self.test = test
        self.n  = 128
        self._base_dir = base_dir

        self.PATCH_SIZE = 64
        # the number of patches per row/column in the
        # image containing all the patches
        self.PATCHES_PER_ROW = 16
        self.PATCHES_PER_IMAGE = self.PATCHES_PER_ROW ** 2
        # the loaded patches
        self._data = dict()

        self.num_trial = 1000000

        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def num_train_patch(self):
        return self._num_train_patch

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_1(self):
        return self._index_1

    @property
    def index_2(self):
        return self._index_2

    def _get_data(self):
        return self._data

    def _get_patches(self):
        return self._get_data()['patches']

    def _get_matches(self):
        return self._get_data()['matches']

    def _get_labels(self):
        return self._get_data()['labels']


    def load_by_name(self, name, patch_size=32, num_channels=1, debug=True):
        assert os.path.exists(os.path.join(self._base_dir, name)) == True, \
            "The dataset directory doesn't exist: %s" % name
        
        img_files = self._load_image_fnames(self._base_dir, name)
        patches = self._load_patches(img_files, name, patch_size, num_channels)
        labels = self._load_labels(self._base_dir, name)
        # load the labels
        self._data['patches'] = patches[0:min(len(labels), len(patches))]
        matches = self._load_matches(self._base_dir, name)
        self._data['labels']  =  labels[0:min(len(labels), len(patches))]
        self._data['matches'] = matches

        # debug info after loading
        if debug:
            print('-- Dataset loaded:    %s' % name)
            print('-- Number of images:  %s' % len(img_files))
            print('-- Number of patches: %s' % len(self._data['patches']))
            print('-- Number of labels:  %s' % len(self._data['labels']))
            print('-- Number of ulabels: %s' % len(np.unique(labels)))
            print('-- Number of matches: %s' % len(matches))

    def _load_matches(self, base_dir, name) :
        fname = os.path.join(base_dir, name, 'm50_100000_100000_0.txt')
        assert os.path.isfile(fname), 'Not a file: %s' % file
        # read file and keep only 3D point ID and 1 if is the same, otherwise 0

        matches = []
        with open(fname, 'r') as f :
            for line in f :
                l = line.split()
                matches.append([int(l[0]), int(l[3]), int(l[1] == l[4])])

        return np.asarray(matches)

    def _load_image_fnames(self, base_dir, dir_name) :
        files = []
        # find those files with the specified extension
        dataset_dir = os.path.join(base_dir, dir_name)
        for file in os.listdir(dataset_dir) :
            if file.endswith('bmp') :
                files.append(os.path.join(dataset_dir, file))
        return sorted(files)

    def _load_patches(self, img_files, name, patch_size, num_channels) :
        patches_all = []
        img_files = img_files[0:self.n] if self.test else img_files
        # load patches
        pbar = tqdm(img_files)
        for file in pbar:
            pbar.set_description('Loading dataset %s' % name)
            # pick file name
            assert os.path.isfile(file), 'Not a file: %s' % file
            img = cv2.imread(file)[:, :, 0] / 255.
            patches_row = np.split(img, self.PATCHES_PER_ROW, axis=0)
            for row in patches_row:
                patches = np.split(row, self.PATCHES_PER_ROW, axis=1)
                for patch in patches:
                    # resize the patch
                    if patch_size != 64:
                        patch = cv2.resize(patch, (patch_size, patch_size))
                    # convert to tensor [w x h x d]
                    patch_tensor = patch.reshape(patch_size, patch_size, num_channels)
                    
                    patches_all.append(patch_tensor)

        return np.asarray(patches_all) if not self.test \
            else np.asarray(patches_all[0:self.n])

    def _load_labels(self, base_dir, dir_name):
        info_fname = os.path.join(base_dir, dir_name, 'info.txt')
        assert os.path.isfile(info_fname), 'Not a file: %s' % file

        labels = []
        with open(info_fname, 'r') as f:
            for line in f:
                labels.append(int(line.split()[0]))
        return np.asarray(labels) if not self.test \
            else np.asarray(labels[0:self.n])

    def _create_indices(self, labels):
        old = labels[0]
        indices = dict()
        indices[old] = 0
        for x in xrange(len(labels) - 1):
            new = labels[x + 1]
            if old != new:
                indices[new] = x + 1
            old = new
        return indices

    def _compute_mean_and_std(self, patches):
        assert len(patches) > 0, 'Patches list is empty!'
        # compute the mean
        mean = np.mean(patches)
        # compute the standard deviation
        std = np.std(patches)
        return mean, std

    def normalize_data(self, mean, std):
        #pbar = tqdm(self._data['patches'])
        pbar = tqdm(xrange(self._data['patches'].shape[0]))
        for i in pbar:
            pbar.set_description('Normalizing data')
            self._data['patches'][i] = (self._data['patches'][i] - mean) / std

    def generate_stats(self):
        #print('-- Computing dataset mean: %s ...' % name)
        # compute the mean and std of all patches
        patches = self._get_patches()
        mean, std = self._compute_mean_and_std(patches)
        print('-- Computing dataset mean:  ... OK')
        print('-- Mean: %s' % mean)
        print('-- Std : %s' % std)
        return mean, std

    def prune(self, min=2):
        labels = self._get_labels()
        ids, labels = self._prune(labels, min)
        return ids, labels

    def _prune(self, labels, min):
        # count the number of labels
        c = collections.Counter(labels)
        # create a list with globals indices
        ids = range(len(labels))
        # remove ocurrences
        ids, labels = self._rename_and_prune(labels, ids, c, min)
        return np.asarray(ids), np.asarray(labels)

    def _rename_and_prune(self, labels, ids, c, min):
        count, x = 0, 0
        labels_new, ids_new = [[] for _ in range(2)]
        while x < len(labels):
            num = c[labels[x]]
            if num >= min:
                for i in xrange(num):
                    labels_new.append(count)
                    ids_new.append(ids[x + i])
                count += 1
            x += num
        return ids_new, labels_new

    def generate_samples_1(self, n_samples):
        self._index_1 = self._get_matches()[:,0]
        self._index_2 = self._get_matches()[:,1]
        self._train_label = self._get_matches()[:,2]
        self._num_train_patch = self._train_label.shape[0]
       
    def generate_samples(self, n_samples):
        # retrieve loaded patches and labels
        labels = self._get_labels()
        # group labels in order to have O(1) search
        count = collections.Counter(labels)
        # index the labels in order to have O(1) search
        indices = self._create_indices(labels)
        # range for the sampling
        labels_size = len(labels) - 1
        # triplets ids
        self._index_1 = []
        self._index_2 = []
        self._train_label = []
        # generate the triplets
        pbar = tqdm(xrange(n_samples))

        for x in pbar:
            pbar.set_description('Generating positive')
            idx = random.randint(0, labels_size)
            num_samples = count[labels[idx]]
            begin_positives = indices[labels[idx]]

            offset_a, offset_p = random.sample(xrange(num_samples), 2)
            while offset_a == offset_p:
                offset_a, offset_p = random.sample(xrange(num_samples), 2)
                
            idx_a = begin_positives + offset_a
            idx_p = begin_positives + offset_p
            self._index_1.append(idx_a)
            self._index_2.append(idx_p)
            self._train_label.append(1)
            idx_n = random.randint(0, labels_size)
            while labels[idx_n] == labels[idx_a] and \
                            labels[idx_n] == labels[idx_p]:
                idx_n = random.randint(0, labels_size)
            self._index_1.append(idx_a)
            self._index_2.append(idx_n)
            self._train_label.append(0)

        self._index_1 = np.array(self._index_1)
        self._index_2 = np.array(self._index_2)
        self._train_label = np.array(self._train_label)

        temp_index = np.arange(self._train_label.shape[0])
        self._num_train_patch = self._train_label.shape[0]

        np.random.shuffle(temp_index)
        self._train_label = self._train_label[temp_index]
        self._index_1 = self._index_1[temp_index]
        self._index_2 = self._index_2[temp_index]

    def generate_triplet(self, n_samples):
        # retrieve loaded patches and labels
        labels = self._get_labels()
        # group labels in order to have O(1) search
        count = collections.Counter(labels)
        # index the labels in order to have O(1) search
        indices = self._create_indices(labels)
        # range for the sampling
        labels_size = len(labels) - 1
        # triplets ids
        self._index_1 = []
        self._index_2 = []
        self._index_3 = []
        # generate the triplets
        pbar = tqdm(xrange(n_samples))

        for x in pbar:
            pbar.set_description('Generating triplets')
            idx = random.randint(0, labels_size)
            num_samples = count[labels[idx]]
            begin_positives = indices[labels[idx]]

            offset_a, offset_p = random.sample(xrange(num_samples), 2)
            while offset_a == offset_p:
                offset_a, offset_p = random.sample(xrange(num_samples), 2)
            idx_a = begin_positives + offset_a
            idx_p = begin_positives + offset_p
            self._index_1.append(idx_a)
            self._index_2.append(idx_p)
            idx_n = random.randint(0, labels_size)
            while labels[idx_n] == labels[idx_a] and \
                            labels[idx_n] == labels[idx_p]:
                idx_n = random.randint(0, labels_size)
            self._index_3.append(idx_n)

        self._index_1 = np.array(self._index_1)
        self._index_2 = np.array(self._index_2)
        self._index_3 = np.array(self._index_3)

        temp_index = np.arange(self._index_1.shape[0])
        self._num_train_patch = self._index_1.shape[0]

        np.random.shuffle(temp_index)
        self._index_1 = self._index_1[temp_index]
        self._index_2 = self._index_2[temp_index]
        self._index_3 = self._index_3[temp_index]

    def generate_structured(self, n_samples):
        # retrieve loaded patches and labels
        labels = self._get_labels()
        # group labels in order to have O(1) search
        count = collections.Counter(labels)
        # index the labels in order to have O(1) search
        indices = self._create_indices(labels)
        # range for the sampling
        labels_size = len(labels) - 1
        # structured ids
        self._index_1 = []
        self._index_2 = []
        self._train_label = []
        # generate batches
        pbar = tqdm(xrange(n_samples))

        for x in pbar:
            pbar.set_description('Generating structured data')
            idx = random.randint(0, labels_size)
            num_samples = count[labels[idx]]
            begin_positives = indices[labels[idx]]

            offset_a, offset_p = random.sample(xrange(num_samples), 2)
            while offset_a == offset_p:
                offset_a, offset_p = random.sample(xrange(num_samples), 2)
            idx_a = begin_positives + offset_a
            idx_p = begin_positives + offset_p
            self._index_1.append(idx_a)
            self._index_1.append(idx_p)
            self._train_label.append(labels[idx_a])
            self._train_label.append(labels[idx_p])
            
            idx_n = random.randint(0, labels_size)

            while labels[idx_n] == labels[idx_a] and \
                            labels[idx_n] == labels[idx_p]:
                idx_n = random.randint(0, labels_size)
            self._index_2.append(idx_n)
            idx_n = random.randint(0, labels_size)

            while labels[idx_n] == labels[idx_a] and \
                            labels[idx_n] == labels[idx_p]:
                idx_n = random.randint(0, labels_size)
            self._index_2.append(idx_n)
        
        #do not use shuffle
        self._index_1 = np.array(self._index_1)
        self._index_2 = np.array(self._index_2)
        self._train_label = np.array(self._train_label)
        self._num_train_patch = self._index_1.shape[0]
        
    def generate_npair(self, n_samples):
        # retrieve loaded patches and labels
        labels = self._get_labels()
        # group labels in order to have O(1) search
        count = collections.Counter(labels)
        # index the labels in order to have O(1) search
        indices = self._create_indices(labels)
        # range for the sampling
        labels_size = len(labels) - 1
        # triplets ids
        self._index_1 = []
        self._index_2 = []
        self._index_3 = []
        self._train_label = []
        # generate the triplets
        pbar = tqdm(xrange(n_samples))

        for x in pbar:
            pbar.set_description('Generating npair triplets')
            idx = random.randint(0, labels_size)
            num_samples = count[labels[idx]]
            begin_positives = indices[labels[idx]]

            offset_a, offset_p = random.sample(xrange(num_samples), 2)
            
            while offset_a == offset_p:
                offset_a, offset_p = random.sample(xrange(num_samples), 2)
            idx_a = begin_positives + offset_a
            idx_p = begin_positives + offset_p
            self._index_1.append(idx_a)
            self._index_2.append(idx_p)
            self._train_label.append(labels[idx])
            idx_n = random.randint(0, labels_size)

            while labels[idx_n] == labels[idx_a] and \
                            labels[idx_n] == labels[idx_p]:
                idx_n = random.randint(0, labels_size)
            self._index_3.append(idx_n)

        self._index_1 = np.array(self._index_1)
        self._index_2 = np.array(self._index_2)
        self._index_3 = np.array(self._index_3)
        self._train_label = np.array(self._train_label)

        temp_index = np.arange(self._index_1.shape[0])
        self._num_train_patch = self._index_1.shape[0]

        np.random.shuffle(temp_index)
        self._index_1 = self._index_1[temp_index]
        self._index_2 = self._index_2[temp_index]
        self._index_3 = self._index_3[temp_index]
        self._train_label = self._train_label[temp_index]

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
            self._epochs_completed += 1
        # Go to the next epoch
        if start + batch_size > self._num_train_patch:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = 0
            temp_index = np.arange(self._train_label.shape[0])

            np.random.shuffle(temp_index)
            self._train_label = self._train_label[temp_index]
            self._index_1 = self._index_1[temp_index]
            self._index_2 = self._index_2[temp_index]

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self._index_1[start:end], \
            self._index_2[start:end], \
            self._train_label[start:end]

    def next_batch_triplet(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
            self._epochs_completed += 1
        # Go to the next epoch
        if start + batch_size > self._num_train_patch:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = 0
            temp_index = np.arange(self._index_1.shape[0])

            np.random.shuffle(temp_index)
            self._index_1 = self._index_1[temp_index]
            self._index_2 = self._index_2[temp_index]
            self._index_3 = self._index_3[temp_index]

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        return self._index_1[start:end], \
            self._index_2[start:end], \
            self._index_3[start:end]

    def next_batch_npair(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        tmp_label = np.array([])
        unique_label = np.unique(tmp_label)
       
        # Skip the batch if it contains duplicated label.
        while unique_label.shape[0] != batch_size:
            start = self._index_in_epoch
            # Shuffle for the first epoch
            if self._epochs_completed == 0 and start == 0:
                self._epochs_completed += 1
            # Go to the next epoch
            if start + batch_size > self._num_train_patch:
                self._epochs_completed += 1
                start = 0
                self._index_in_epoch = 0
                temp_index = np.arange(self._index_1.shape[0])

                np.random.shuffle(temp_index)
                self._train_label = self._train_label[temp_index]
                self._index_1 = self._index_1[temp_index]
                self._index_2 = self._index_2[temp_index]
                self._index_3 = self._index_3[temp_index]

            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            tmp_label = self._train_label[start:end]
            unique_label = np.unique(tmp_label)

        return self._index_1[start:end], \
            self._index_2[start:end], \
            self._index_3[start:end]

    def next_batch_structured(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
            self._epochs_completed += 1
        # Go to the next epoch
        if start + batch_size > self._num_train_patch:
            self._epochs_completed += 1
            start = 0
            self._index_in_epoch = 0
            temp_index = np.arange(self._index_1.shape[0])

        self._index_in_epoch += batch_size
        end = self._index_in_epoch

        label = self._train_label[start:end].reshape((batch_size,1))
        tmp_label = np.matlib.repmat(label,1,batch_size)
        same_label = tmp_label == np.transpose(tmp_label)
        same_label = same_label.astype(np.float32)
        different_label = 1.0 - same_label 

        for i in range(batch_size):
            same_label[i,i] = 0.0

        return self._index_1[start:end], \
            self._index_2[start:end], \
            same_label, \
            different_label
