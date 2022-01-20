from torch.utils.data import Dataset


class TextData(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data.iloc[index]['content']
