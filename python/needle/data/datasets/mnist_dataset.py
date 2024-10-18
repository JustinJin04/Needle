from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename,"rb") as image_file:
            magic_num,num_images,num_rows,num_cols=struct.unpack('>4i',image_file.read(16))
            assert(magic_num==2051)
            tot_pixels=num_rows*num_cols
            
            self.X=np.array(struct.unpack(f"{num_images*tot_pixels}B",image_file.read(num_images*tot_pixels)),dtype=np.float32)
            self.X=self.X.reshape(num_images,tot_pixels)
            self.X-=np.min(self.X)
            self.X/=np.max(self.X)
        
        with gzip.open(label_filename,"rb") as label_file:
            magic_num,num_labels=struct.unpack('>2i',label_file.read(8))
            assert(magic_num==2049)
            
            self.y=np.array(struct.unpack(f"{num_labels}B",label_file.read(num_labels)),dtype=np.uint8)
        
        self.transforms = transforms

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X = self.X[index]
        y = self.y[index]

        if len(X.shape) == 1:
            return self.apply_transforms(X.reshape(28,28,1)).reshape((28*28,)), y
        else:
            imgs = []
            for img in X:
                imgs.append(self.apply_transforms(img.reshape(28,28,1)).reshape((28*28,)))
            return np.stack(imgs,axis=0), y
            
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION