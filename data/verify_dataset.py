from data.image_folder import make_dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class VerifyDataset(Dataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, data_root, max_dataset_size=float('inf')):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.A_paths = sorted(make_dataset(data_root, max_dataset_size))
        self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
