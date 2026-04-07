# ============================================================================
# File: util.py
# Date: 2026-03-27
# Author: TA
# Description: Utility functions to process BoW features and KNN classifier.
# ============================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from scipy.spatial.distance import cdist
from collections import Counter

CAT = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
       'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
       'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

CAT2ID = {v: k for k, v in enumerate(CAT)}

########################################
###### FEATURE UTILS              ######
###### use TINY_IMAGE as features ######
########################################

###### Step 1-a
def get_tiny_images(img_paths: list):
    tiny_img_feats = []
    for path in tqdm(img_paths, desc="Getting tiny images"):
        img = Image.open(path).convert('L')
        img = img.resize((16, 16))
        feat = np.array(img, dtype=np.float32).flatten()
        feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-6)
        tiny_img_feats.append(feat)
    return np.array(tiny_img_feats)

#########################################
###### FEATURE UTILS               ######
###### use BAG_OF_SIFT as features ######
#########################################

###### Step 1-b-1
def build_vocabulary(
        img_paths: list, 
        vocab_size: int = 400
    ):
    all_sift_feats = []
    for path in tqdm(img_paths, desc="Building vocabulary"):
        img = Image.open(path).convert('L')
        img = np.array(img, dtype=np.float32)
        step = [15, 15]
        _, descs = dsift(img, step=step, fast=True)
        if descs is not None and descs.shape[0] > 0:
            idxs = np.random.choice(descs.shape[0], size=min(100, descs.shape[0]), replace=False)
            all_sift_feats.append(descs[idxs])
    
    all_sift_feats = np.vstack(all_sift_feats).astype(np.float32)
    vocab = kmeans(all_sift_feats, vocab_size, initialization="PLUSPLUS")
    return vocab

###### Step 1-b-2
def get_bags_of_sifts(
        img_paths: list,
        vocab: np.array
    ):
    img_feats = []
    for path in tqdm(img_paths, desc="Getting bag of sifts"):
        img = Image.open(path).convert('L')
        img = np.array(img, dtype=np.float32)
        _, descs = dsift(img, step=[5, 5], fast=True)
        if descs is not None and descs.shape[0] > 0:
            descs = descs.astype(np.float32)
            dists = cdist(descs, vocab)
            closest = np.argmin(dists, axis=1)
            hist, _ = np.histogram(closest, bins=np.arange(vocab.shape[0] + 1))
            hist = hist.astype(np.float32)
            hist /= (np.sum(hist) + 1e-6)
            img_feats.append(hist)
        else:
            img_feats.append(np.zeros(vocab.shape[0], dtype=np.float32))
    return np.array(img_feats)

################################################
###### CLASSIFIER UTILS                   ######
###### use NEAREST_NEIGHBOR as classifier ######
################################################

###### Step 2
def nearest_neighbor_classify(
        train_img_feats: np.array,
        train_labels: list,
        test_img_feats: list
    ):
    test_predicts = []
    test_img_feats = np.array(test_img_feats)
    train_img_feats = np.array(train_img_feats)
    dists = cdist(test_img_feats, train_img_feats, metric='minkowski', p=1)
    
    k = 5
    for i in range(test_img_feats.shape[0]):
        closest_idxs = np.argsort(dists[i])[:k]
        closest_labels = [train_labels[idx] for idx in closest_idxs]
        counts = Counter(closest_labels)
        best_label = counts.most_common(1)[0][0]
        test_predicts.append(best_label)
        
    return test_predicts
