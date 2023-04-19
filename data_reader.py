import torch
from torch.utils.data import DataLoader, Dataset
import os
import torchvision


class VideoLoader(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.videos_path = [os.path.join(path, i) for i in os.listdir(path)]
        self.transform = transform

    def __len__(self):
        return len(self.videos_path)

    def __getitem__(self, idx):
        stream = 'video'
        video_reader = torchvision.io.VideoReader(self.videos_path[idx], stream=stream)
        frames = [frame['data'] for frame in video_reader]
        frames = torch.stack(frames)
        if self.transform:
            frames = self.transform(frames)
        return frames
