import torch
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms.functional as F


class Dataset(Dataset):
    def __init__(self, source, target, config):
        self.source = source
        self.target = target
        if config.data_augmentation:
            for i in range(5):
                ids = torch.randperm(source.shape[0])
                sep = int(source.shape[0] * 0.2)
                sep_ids = ids[:sep]
                random_source = source[sep_ids]
                random_target = target[sep_ids]
                if int(torch.randint(low=0, high=100, size=[1])) > 50:
                    angles = [-90,90,180]
                    angle = angles[int(torch.randint(low=0, high= 3, size=[1]))]
                    random_source = F.rotate(random_source,angle)
                    random_target = F.rotate(random_target, angle)
                if int(torch.randint(low=0, high=100, size=[1])) > 50:
                    random_source = F.vflip(random_source)
                    random_target = F.vflip(random_target)
                if int(torch.randint(low=0, high=100, size=[1])) > 50:
                    random_source = F.hflip(random_source)
                    random_target = F.hflip(random_target)
                self.source = torch.cat([self.source,random_source])
                self.target = torch.cat([self.target,random_target])

        self.source = self.source.div(255)
        self.target = self.target.div(255)
        if config.normalize:
            mean_c1 = self.source[:, :, :, 0].float().mean()
            mean_c2 = self.source[:, :, :, 1].float().mean()
            mean_c3 = self.source[:, :, :, 2].float().mean()
            std_c1 = self.source[:, :, :, 0].float().std()
            std_c2 = self.source[:, :, :, 1].float().std()
            std_c3 = self.source[:, :, :, 2].float().std()
            self.transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((mean_c1,mean_c2,mean_c3), (std_c1, std_c2, std_c3))])
        else:
            self.transform = None

    def __getitem__(self, i):
        img, target = self.source[i], self.target[i]
        if self.transform is not None:
            img = self.transform(img)



        return img,target

    def __len__(self):
        return self.source.shape[0]
