import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        def unpickle(file):
            with open(file, "rb") as fo:
                dict = pickle.load(fo,encoding='bytes')
            return dict

        # TODO: dtype=float32?
        
        self.X = None
        self.y = None
        if train:
            for i in range(1,6,1):
                data_batch_filename = os.path.join(base_folder, f"data_batch_{i}")
                dict = unpickle(data_batch_filename)
                if self.X is None:
                    self.X = dict[b'data']
                    self.y = dict[b'labels']
                else:
                    self.X = np.concatenate([self.X, dict[b'data']], axis=0)
                    self.y = np.concatenate([self.y, dict[b'labels']], axis=0)
        else:
            test_batch_filename = os.path.join(base_folder, "test_batch")
            dict = unpickle(test_batch_filename)
            self.X = dict[b'data']
            self.y = dict[b'labels']

        self.X = self.X / 255
        self.transforms = transforms
                
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        assert self.transforms == None, "TODO: now transform is diabled"
        X = self.X[index]
        y = self.y[index]

        if len(X.shape) == 1:
            return X.reshape(3,32,32), y
        else:
            return X.reshape((-1, 3, 32, 32)), y

        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
