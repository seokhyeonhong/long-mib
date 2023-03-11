import os
import pickle
import numpy as np

from pymovis.motion import BVH
from pymovis.ops import rotation
from pymovis.utils import util

from utility.config import Config

""" Load BVH files and convert to Motion objects """
def load_motions(config):
    train_files, test_files = [], []

    for file in sorted(os.listdir(config.trainset_dir)):
        if file.endswith(".bvh"):
            train_files.append(os.path.join(config.trainset_dir, file))
    
    for file in sorted(os.listdir(config.testset_dir)):
        if file.endswith(".bvh"):
            test_files.append(os.path.join(config.testset_dir, file))
    
    train_motions = BVH.load_parallel(train_files, v_forward=config.v_forward, v_up=config.v_up, to_meter=0.01)
    test_motions  = BVH.load_parallel(test_files, v_forward=config.v_forward, v_up=config.v_up, to_meter=0.01)

    return train_motions, test_motions

""" Sliding window """
def get_windows_parallel(motions, window_length, window_offset, align_at):
    windows = []
    for ws in util.run_parallel_sync(get_windows, motions, window_length=window_length, window_offset=window_offset, align_at=align_at, desc="Making windows"):
        windows.extend(ws)
    print("\t- Number of windows:", len(windows))
    return windows

def get_windows(motion, window_length, window_offset, align_at):
    windows = []
    for start in range(0, motion.num_frames - window_length, window_offset):
        window = motion.make_window(start, start+window_length)
        window.align_by_frame(align_at, origin_axes="xz")
        windows.append(window)

    return windows

""" Feature extraction """
def get_features_parallel(windows):
    features = []
    for fs in util.run_parallel_sync(get_features, windows, desc="Extracting features"):
        features.append(fs)
    
    features = np.stack(features, axis=0).astype(np.float32)
    print("\t- Features shape:", features.shape)
    return features

def get_features(window):
    local_R = np.stack([pose.local_R for pose in window.poses], axis=0)
    root_p  = np.stack([pose.root_p for pose in window.poses], axis=0)

    local_R6 = rotation.R_to_R6(local_R).reshape(len(window), -1)
    root_p   = root_p.reshape(len(window), -1)
    feature = np.concatenate([local_R6, root_p], axis=-1).astype(np.float32)

    return feature

""" Save """
def save_skeleton(skeleton, dataset_dir):
    save_path = os.path.join(dataset_dir, "skeleton.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(skeleton, f)

def save_features(features, npy_path):
    np.save(npy_path, features)

def main():
    config = Config.load("configs/config.json")
    train_motions, test_motions = load_motions(config)

    # feature extraction
    train_windows  = get_windows_parallel(train_motions, config.window_length, config.window_offset, config.context_frames - 1)
    train_features = get_features_parallel(train_windows)

    test_windows  = get_windows_parallel(test_motions, config.window_length, config.window_offset, config.context_frames - 1)
    test_features = get_features_parallel(test_windows)

    # save
    save_skeleton(train_motions[0].skeleton, config.dataset_dir)
    save_features(train_features, config.trainset_npy)
    save_features(test_features, config.testset_npy)

if __name__ == "__main__":
    main()