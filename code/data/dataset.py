import numpy as np
import torch
from skimage import io
from torch.utils.data.sampler import SubsetRandomSampler


class ImagesDataset:
    def __init__(self, list_csv, train=True, transform=None,
                 image_path='../dataset/state-farm-distracted-driver-detection/imgs/train/'):
        '''
        Args:
            path (string): path to images folder
            answers (DataFrame): data of keypoints in pandas dataframe format.
            train (Boolean) : True for train data with keypoints, default is True
            transform (callable, optional): Optional transform to be applied on
            sample
        '''
        self.list_csv = list_csv
        self.train = train
        self.transform = transform
        self.image_path = image_path

    def __len__(self):
        return len(self.list_csv)

    def __getitem__(self, ind):
        cl = self.list_csv['classname'][ind]
        name = self.list_csv['img'][ind]
        # print(cl, name, self.list_csv.iloc[cl])
        mode = 'train' if self.train else 'val'

        image = io.imread(self.image_path + cl + '/' + name)
        sample = {'image': image, 'answer': cl}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Normalize(object):
    '''Normalize input images'''

    def __call__(self, sample):
        image, answer = sample['image'], sample['answer']
        return {'image': image / 255.,  # scale to [0, 1]
                'answer': answer}


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''

    def __call__(self, sample):
        image, answer = sample['image'], sample['answer']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.T
        image = torch.from_numpy(image)
        return {'image': image, 'answer': answer}

def prepare_train_valid_loaders(trainset, valid_size=0.2,
                                batch_size=128):
    '''
    Split trainset data and prepare DataLoader for training and validation

    Args:
        trainset (Dataset): data
        valid_size (float): validation size, defalut=0.2
        batch_size (int) : batch size, default=128
    '''

    # obtain training indices that will be used for validation
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               sampler=valid_sampler)

    return train_loader, valid_loader
