
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torchvision.transforms as T

class SegmentationSample(Dataset):
    def __init__(self, root_dir, image_file, device):
        self.image_file = os.path.join(root_dir, image_file)
        self.image = Image.open(self.image_file)

        self.img_width, self.img_height = self.image.size[:]

        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.preprocessing = T.Compose([
            T.Resize(450),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.unload_tensor = T.ToPILImage()

        # Output returned: processed tensor with an additional dim for the batch:
        self.processed_image = self.preprocessing(self.image)
        self.processed_image = self.processed_image.unsqueeze(0).to(self.device)

    def __getitem__(self, item):
        return self.processed_image