"""
2024-10-09 MM: copied from Brendan's(?) implementation in our dinov2 fork.
"""

import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np

import logging

logger = logging.getLogger(__name__)

def default_loader(path):
    return Image.open(path).convert('RGB')

class CSVDataset(Dataset):

    def __init__(self, rootdir, csvpath, loader=default_loader, transform=None, target_transform=None):
        """
        Dataset which loads labels and paths from a CSV file.
        Requires a CSV file with columns 'path' and 'label', where 'path' is the path to the image and 'label' is the class

        :param rootdir: Root directory of images, paths in csv are relative to this
        :param csvpath: CSV File with class strings and image paths
        :param transforms: Img transforms
        :param target_transform: Target transforms, by default it maps the label to an index
        :param labelmap: Optional dictionary mapping labels to indices
        :param loader: Function to load an image given a path
        """
        super().__init__()
        self.rootdir = Path(rootdir)
        assert self.rootdir.is_dir(), f"{self.rootdir} is not a directory"
        self.csvpath = Path(csvpath)
        self.data = pd.read_csv(csvpath)
        self.loader = loader
        self.transform = transform
        if target_transform is not None:
            logger.warning(f"Got a target-transform of {target_transform} for CSVDataset, but this is not implemented as there are no targets")

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        _repr_indent = 4
        head = self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.rootdir}")
        body.append(f"Paths csv: {self.csvpath}")
        body.append(f"Loader: {self.loader.__module__ + '.' + self.loader.__name__}")
        if hasattr(self, "transform") and self.transform is not None:
            body.append(f"Transform:")
            for line in repr(self.transform).split("\n"):
                body.append(" " * _repr_indent + line)
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def __getitem__(self, item):
        row = self.data.iloc[item]

        #logger.info(f"Getting item {item} at {row.path}")
        try:
            img = self.loader(self.rootdir / row.path)
            #logger.info(f"Got image data: {np.array(img)[0:2, 0:2, 0:2]}")
        except Exception as ex:
            logger.error(f"Could not open item {row.path}")
            raise ex
    
        if self.transform:
            #logger.info(f"Executing transforms")
            try:
                img = self.transform(img)
            except Exception as ex:
                logger.error(f"Error transforming image #{item} : {row.path} : {ex}")
                raise ex
        return img, None
