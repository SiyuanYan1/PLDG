import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import cv2
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    "SKIN",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    "DG_Dataset",
    "Latent_DR_Dataset"
]

from torch.utils.data import Dataset
import sys
import os
from torchvision import transforms
from torchvision.datasets.folder import make_dataset, default_loader
import numpy as np

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class Latent_DR_Dataset(Dataset):
    def __init__(self, root_dir, domain, split, get_domain_label=False, get_cluster=False, color_jitter=True,
                 min_scale=0.8):
        self.root_dir = root_dir
        self.domain = domain
        self.split = split
        self.get_domain_label = get_domain_label
        self.get_cluster = get_cluster
        self.color_jitter = color_jitter
        self.min_scale = min_scale
        self.set_transform(self.split)
        self.loader = default_loader
        self.N_STEPS = 5001  # Default, subclasses may override
        self.CHECKPOINT_FREQ = 100  # Default, subclasses may override
        self.N_WORKERS = 8  # Default, subclasses may override
        self.ENVIRONMENTS = None  # Subclasses should override
        self.INPUT_SHAPE = None  # Subclasses should override
        self.CHECKPOINT_FREQ = 1
        self.input_shape = (3, 224, 224,)
        # self.num_domain=2

        self.num_classes = 5

        self.load_dataset()

    def __len__(self):
        return len(self.images)
    # import torchsnooper
    # @torchsnooper.snoop()
    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        image = self.loader(path)
        image = self.transform(image)
        output = [image, target]

        if self.get_domain_label:
            domain = np.copy(self.domains[index])
            domain = np.int64(domain)
            output.append(domain)
        if self.get_cluster:
            cluster = np.copy(self.clusters[index])
            cluster = np.int64(cluster)
            output.append(cluster)

        return tuple(output)  # (image, label,domain_label,pseudo domain label)

    import torchsnooper
    # @torchsnooper.snoop()
    def find_classes(self, dir_name):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    # import torchsnooper
    # @torchsnooper.snoop()
    def load_dataset(self):
        total_samples = []
        self.domains = np.zeros(0)
        # classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        #  class_to_idx = {'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}
        classes, class_to_idx = self.find_classes(self.root_dir + self.domain[0] + '/')
        self.num_class = len(classes)
        messidor_idx={"0":0,"1":1,"2":2,"3":3}
        # self.domain=['art_painting', 'cartoon', 'sketch']
        for i, item in enumerate(self.domain):
            path = self.root_dir + item + '/'
            # samples= [(img_path,class_id),(,),...]
            # samples = [('/mount/neuron/Lamborghini/dir/pythonProject/d.../PACS/kfold/art_painting/person/pic_577.jpg', 6)]
            # all samples for one domain
            if "Messidor-1" in path:
                # Call make_dataset() if the folder exists
                samples = make_dataset(path, messidor_idx, IMG_EXTENSIONS)
                total_samples.extend(samples)
                self.domains = np.append(self.domains, np.ones(len(samples)) * i)
            else:
                samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
                total_samples.extend(samples)
                self.domains = np.append(self.domains, np.ones(len(samples)) * i)
        # self.domains=domain label for each image_data_generator
        self.clusters = np.zeros(len(self.domains), dtype=np.int64)
        # total_samples=[(),(),...]
        self.images = [s[0] for s in total_samples]
        self.labels = [s[1] for s in total_samples]

    def set_cluster(self, cluster_list):
        if len(cluster_list) != len(self.images):
            raise ValueError("The length of cluster_list must to be same as self.images")
        else:
            self.clusters = cluster_list

    def set_domain(self, domain_list):
        if len(domain_list) != len(self.images):
            raise ValueError("The length of domain_list must to be same as self.images")
        else:
            self.domains = domain_list

    def set_transform(self, split):
        if split == 'train':
            if self.color_jitter:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                    transforms.RandomRotation(45),
                    transforms.ColorJitter(hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                    transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        elif split == 'val' or split == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            raise Exception('Split must be train or val or test!!')

class DG_Dataset(Dataset):
    def __init__(self, root_dir, domain, split, get_domain_label=False, get_cluster=False, color_jitter=True,
                 min_scale=0.8):
        self.root_dir = root_dir
        self.domain = domain
        self.split = split
        self.get_domain_label = get_domain_label
        self.get_cluster = get_cluster
        self.color_jitter = color_jitter
        self.min_scale = min_scale
        self.set_transform(self.split)
        self.loader = default_loader
        self.N_STEPS = 5001  # Default, subclasses may override
        self.CHECKPOINT_FREQ = 100  # Default, subclasses may override
        self.N_WORKERS = 8  # Default, subclasses may override
        self.ENVIRONMENTS = None  # Subclasses should override
        self.INPUT_SHAPE = None  # Subclasses should override
        self.CHECKPOINT_FREQ = 1
        self.input_shape = (3, 224, 224,)
        self.num_domain=5

        self.num_classes = 2

        self.load_dataset()

    def __len__(self):
        return len(self.images)
    # import torchsnooper
    # @torchsnooper.snoop()
    def __getitem__(self, index):
        path, target = self.images[index], self.labels[index]
        image = self.loader(path)
        image = self.transform(image)
        output = [image, target]

        if self.get_domain_label:
            domain = np.copy(self.domains[index])
            domain = np.int64(domain)
            output.append(domain)
        if self.get_cluster:
            cluster = np.copy(self.clusters[index])
            cluster = np.int64(cluster)
            output.append(cluster)

        return tuple(output)  # (image, label,domain_label,pseudo domain label)

    import torchsnooper
    # @torchsnooper.snoop()
    def find_classes(self, dir_name):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir_name) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    # import torchsnooper
    # @torchsnooper.snoop()
    def load_dataset(self):
        total_samples = []
        self.domains = np.zeros(0)
        # classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        #  class_to_idx = {'dog': 0, 'elephant': 1, 'giraffe': 2, 'guitar': 3, 'horse': 4, 'house': 5, 'person': 6}
        classes, class_to_idx = self.find_classes(self.root_dir + self.domain[0] + '/')
        self.num_class = len(classes)
        # self.domain=['art_painting', 'cartoon', 'sketch']
        for i, item in enumerate(self.domain):
            path = self.root_dir + item + '/'
            # samples= [(img_path,class_id),(,),...]
            # samples = [('/mount/neuron/Lamborghini/dir/pythonProject/d.../PACS/kfold/art_painting/person/pic_577.jpg', 6)]
            # all samples for one domain
            # print(path)
            # print('ccccccccccccccc',class_to_idx)
            samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
            total_samples.extend(samples)
            self.domains = np.append(self.domains, np.ones(len(samples)) * i)
        # self.domains=domain label for each image_data_generator
        self.clusters = np.zeros(len(self.domains), dtype=np.int64)
        # total_samples=[(),(),...]
        self.images = [s[0] for s in total_samples]
        self.labels = [s[1] for s in total_samples]

    def set_cluster(self, cluster_list):
        if len(cluster_list) != len(self.images):
            raise ValueError("The length of cluster_list must to be same as self.images")
        else:
            self.clusters = cluster_list

    def set_domain(self, domain_list):
        if len(domain_list) != len(self.images):
            raise ValueError("The length of domain_list must to be same as self.images")
        else:
            self.domains = domain_list

    def set_transform(self, split):
        if split == 'train':
            if self.color_jitter:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                    transforms.RandomRotation(45),
                    transforms.ColorJitter(hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                    transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        elif split == 'val' or split == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            raise Exception('Split must be train or val or test!!')

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                           self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
                                                         1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


#
# class RotatedMNIST(MultipleEnvironmentMNIST):
#     ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']
#
#     def __init__(self, root, test_envs, hparams):
#         super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
#                                            self.rotate_dataset, (1, 28, 28,), 10)
#
#     def rotate_dataset(self, images, labels, angle):
#         rotation = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
#                 interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
#             transforms.ToTensor()])
#
#         x = torch.zeros(len(images), 1, 28, 28)
#         for i in range(len(images)):
#             x[i] = rotation(images[i])
#
#         y = labels.view(-1)
#
#         return TensorDataset(x, y)
# import torchsnooper
# @torchsnooper.snoop()
class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)  # kfold
        print('EEEEEEE',environments)

        augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomRotation(45),
            transforms.ColorJitter(hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.datasets = []

        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                                      transform=env_transform)

            self.datasets.append(env_dataset)
        self.input_shape = (3, 224, 224,)

        self.num_classes = len(self.datasets[-1].classes)

        print("num class: ", self.num_classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/kfold/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SKIN(MultipleEnvironmentImageFolder):
    # CHECKPOINT_FREQ = 56
    CHECKPOINT_FREQ = 1
    ENVIRONMENTS = ["C", "D", "G", "H", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "ISIC2019_train/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class DR(MultipleEnvironmentImageFolder):
    # CHECKPOINT_FREQ = 56
    CHECKPOINT_FREQ = 1
    ENVIRONMENTS = ["A", "E", "M", "N"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "DG_DR_Classification/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    N_WORKERS = 1

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]

    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)

    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)
            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = ["hospital_0", "hospital_1", "hospital_2", "hospital_3",
                    "hospital_4"]

    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = ["region_0", "region_1", "region_2", "region_3",
                    "region_4", "region_5"]

    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame,imfolder):
        """
		Class initialization
		Args:
			df (pd.DataFrame): DataFrame with data description
			imfolder (str): folder with images
			train (bool): flag of whether a training dataset is being initialized or testing one
			transforms: image transformation method to be applied
			meta_features (list): list of features with meta information, such as sex and age

		"""
        self.df = df
        self.imfolder = imfolder

    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image'] + '.jpg')
        x= Image.open(im_path).convert('RGB')
        # meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        meta = None
        x = val_transform(x)
        # if self.train:
        y = self.df.iloc[index]['label']
        return x, y


    def __len__(self):
        return len(self.df)


import os
import os.path
import torch
import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


# TODO: Make target_field optional for unannotated datasets. this is for ood evaluation dataset
class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                 loader=default_loader, transform=None,
                 target_transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 split=None, environment=None, onlylabels=None,
                 subset=None):
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform
        self.target_transform = target_transform
        self.add_extension = add_extension
        self.environment = environment
        self.onlylabels = onlylabels
        self.subset = subset
        self.data = pd.read_csv(csv_file)

        # Environments
        if self.environment is not None:

            all_envs = ["dark_corner", "hair", "gel_border", "gel_bubble", "ruler", "ink", "patches"]
            for env in all_envs:
                if env in self.environment:
                    self.data = self.data[self.data[env] >= 0.6]
                else:
                    self.data = self.data[self.data[env] <= 0.6]
            self.data = self.data.reset_index()

        # Only Label
        if self.onlylabels is not None:
            self.onlylabels = [int(i) for i in self.onlylabels]
            self.data = self.data[self.data[self.target_field].isin(self.onlylabels)]
            self.data = self.data.reset_index()

        # Subset
        if self.subset is not None:
            self.data = self.data[self.data['image'].isin(self.subset)]
            self.data = self.data.reset_index()

        # Split
        if split is not None:
            with open(split, 'r') as f:
                selected_images = f.read().splitlines()
            self.data = self.data[self.data[image_field].isin(selected_images)]
            self.data = self.data.reset_index()

        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(self.data['label'].value_counts())
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls]
                                for cls in self.data['label']]
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]
        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()

        if type(limit) == int:
            limit = (0, limit)
        if type(limit) == tuple:
            self.data = self.data[limit[0]:limit[1]]
            self.data = self.data.reset_index()

        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        # print('Found {} images from {} classes.'.format(len(self.data),
        #                                                 len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.data[self.target_field].value_counts())
            # print("    Class '{}' ({}): {} images.".format(
            #     class_name, idx, n_images[class_name]))         # print("    Class '{}' ({}): {} images.".format(
            #     class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension:
            path = path + self.add_extension

        sample = self.loader(path)
        if self.transform is not None:
            try:
                sample = self.transform(sample)
            except:
                sample = np.array(sample)
                sample = self.transform(image=sample.astype(np.uint8))["image"]

        target = self.class_to_idx[self.data.loc[index, self.target_field]]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data)


import os
import torch
from PIL import Image
from torch.utils.data import Dataset
class FolderImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        for target_class in self.classes:
            class_dir = os.path.join(root_dir, target_class)
            for file_name in os.listdir(class_dir):
                path = os.path.join(class_dir, file_name)
                self.samples.append((path, self.class_to_idx[target_class]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target
