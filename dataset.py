import torch

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, seqs, labels=None, mode='train'):
        self.mode = mode
        # self.imgs = imgs
        self.seqs = seqs
        if self.mode == 'train':
            self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        # img = cv2.imread(self.imgs[i]).astype(np.float32)/255
        # img = np.transpose(img, (2,0,1))
        if self.mode == 'train':
            return {
                # 'img' : torch.tensor(img, dtype=torch.float32),
                'seq': torch.tensor(self.seqs[i], dtype=torch.long),
                'label': torch.tensor(self.labels[i], dtype=torch.float32)
            }
        else:
            return {
                # 'img' : torch.tensor(img, dtype=torch.float32),
                'seq': torch.tensor(self.seqs[i], dtype=torch.long),
            }