import argparse
import os

import imageio
import rasterio
import cv2
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import flooddamagesense.datasets.imutils as imutils


def img_loader(path):
    if '.tif' in path:
        with rasterio.open(path, 'r') as src:
            img = src.read().transpose([1, 2, 0])
    else:
        img = imageio.imread(path)
    return img


def one_hot_encoding(image, num_classes=8):
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    return one_hot


class DamageAssessmentDataset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_epochs=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_epochs is not None:
            self.data_list = self.data_list * max_epochs
            
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, loc_label, clf_label, map_label):
        if aug:
            pre_img, post_img, loc_label, clf_label, map_label = imutils.random_crop(\
                pre_img, post_img, crop_size = self.crop_size, \
                mean=[0.23651549, 0.31761484, -14.57879175, -8.6098158, 0.18514981, 0.26901252, -14.29073382, -8.33534564], \
                loc_label=loc_label, clf_label=clf_label, map_label=map_label)
            loc_label[np.isnan(loc_label)] = 255
            loc_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
            clf_label[np.isnan(clf_label)] = 255
            clf_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
            clf_label[np.isnan(post_img).sum(axis=2) != 0] = 255
            map_label[np.isnan(map_label)] = 255
            map_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
            map_label[np.isnan(post_img).sum(axis=2) != 0] = 255
            pre_img[np.isnan(pre_img)] = 255
            post_img[np.isnan(post_img)] = 255
            pre_img, post_img, loc_label, clf_label, map_label = imutils.random_fliplr(pre_img, post_img, loc_label, clf_label, map_label)
            pre_img, post_img, loc_label, clf_label, map_label = imutils.random_flipud(pre_img, post_img, loc_label, clf_label, map_label)
            pre_img, post_img, loc_label, clf_label, map_label = imutils.random_rot(pre_img, post_img, loc_label, clf_label, map_label)

        pre_img = imutils.normalize_img(pre_img, mean=[0.23651549, 0.31761484, -14.57879175, -8.6098158], std=[0.16280619, 0.20849304, 4.07141682, 3.94773216])
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img, mean=[0.18514981, 0.26901252, -14.29073382, -8.33534564], std=[0.14008107, 0.19767644, 4.21006244, 4.05494136])
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, loc_label, clf_label, map_label

    def __getitem__(self, index):
        img_path = os.path.join(self.dataset_path, self.data_list[index].replace('GT', 'SAR'))
        loc_label_path = os.path.join(self.dataset_path.replace('UrbanSARFloods_v1', 'USBuildingFootprints'), self.data_list[index])
        clf_label_path = os.path.join(self.dataset_path.replace('UrbanSARFloods_v1', 'PDE'), self.data_list[index])
        map_label_path = os.path.join(self.dataset_path, self.data_list[index])
        if os.path.exists(img_path):
            img = self.loader(img_path)
        else:
            img = np.ones((512, 512, 8), dtype=np.uint8) * 255  # Placeholder for missing images
        pre_img = img[:, :, [0, 1, 4, 5]]
        post_img = img[:, :, [2, 3, 6, 7]]
        if os.path.exists(loc_label_path):      
            loc_label = self.loader(loc_label_path)[:,:,0]
        else:
            loc_label = np.ones_like(pre_img[:,:,0]) * 255
        if os.path.exists(clf_label_path):
            clf_label = self.loader(clf_label_path)[:,:,0]
        else:
            clf_label = np.ones_like(pre_img[:,:,0]) * 255
        if os.path.exists(map_label_path):
            map_label = self.loader(map_label_path)[:,:,0]
        else:
            map_label = np.ones_like(pre_img[:,:,0]) * 255
            
        loc_label[np.isnan(loc_label)] = 255
        loc_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
        clf_label[clf_label < 13.66] = 1
        clf_label[(clf_label >= 13.66) & (clf_label < 38.97)] = 2
        clf_label[(clf_label >= 38.97) & (clf_label < 254)] = 3
        clf_label[clf_label == 254] = 0
        clf_label[np.isnan(clf_label)] = 255
        clf_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
        clf_label[np.isnan(post_img).sum(axis=2) != 0] = 255
        map_label[np.isnan(map_label)] = 255
        map_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
        map_label[np.isnan(post_img).sum(axis=2) != 0] = 255
        pre_img[np.isnan(pre_img)] = 255
        post_img[np.isnan(post_img)] = 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, loc_label, clf_label, map_label = self.__transforms(True, pre_img, post_img, loc_label, clf_label, map_label)
        else:
            pre_img, post_img, loc_label, clf_label, map_label = self.__transforms(False, pre_img, post_img, loc_label, clf_label, map_label)
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)
            map_label = np.asarray(map_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, loc_label, clf_label, map_label, data_idx

    def __len__(self):
        return len(self.data_list)

    
class DamageAssessmentFusionDataset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size, max_epochs=None, type='train', data_loader=img_loader):
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_epochs is not None:
            self.data_list = self.data_list * max_epochs
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label):
        if aug:
            pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label = imutils.random_crop(\
                pre_img, post_img, pre_rgb, prior_data, crop_size = self.crop_size, \
                mean=[0.23651549, 0.31761484, -14.57879175, -8.6098158, 0.18514981, 0.26901252, -14.29073382, -8.33534564, 0, 0, 0, 0], \
                loc_label=loc_label, clf_label=clf_label, map_label=map_label)
            prior_data[np.isnan(prior_data)] = 255
            prior_data[np.isnan(pre_img).sum(axis=2) != 0] = 255
            prior_data[np.isnan(post_img).sum(axis=2) != 0] = 255
            loc_label[np.isnan(loc_label)] = 255
            loc_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
            clf_label[np.isnan(clf_label)] = 255
            clf_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
            clf_label[np.isnan(post_img).sum(axis=2) != 0] = 255
            map_label[np.isnan(map_label)] = 255
            map_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
            map_label[np.isnan(post_img).sum(axis=2) != 0] = 255
            pre_img[np.isnan(pre_img)] = 255
            post_img[np.isnan(post_img)] = 255
            pre_rgb[np.isnan(pre_rgb)] = 255
            pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label = imutils.random_fliplr(pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label)
            pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label = imutils.random_flipud(pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label)
            pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label = imutils.random_rot(pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label)

        pre_img = imutils.normalize_img(pre_img, mean=[0.23651549, 0.31761484, -14.57879175, -8.6098158], std=[0.16280619, 0.20849304, 4.07141682, 3.94773216])
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img, mean=[0.18514981, 0.26901252, -14.29073382, -8.33534564], std=[0.14008107, 0.19767644, 4.21006244, 4.05494136])
        post_img = np.transpose(post_img, (2, 0, 1))

        pre_rgb = imutils.normalize_img(pre_rgb)
        pre_rgb = np.transpose(pre_rgb, (2, 0, 1))

        prior_data = imutils.normalize_img(prior_data, mean = [0.007869869], std = [0.0066061574])
        prior_data = np.transpose(prior_data, (2, 0, 1))

        return pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label

    def __getitem__(self, index):
        size = 1280 #self.crop_size *2
        img_path = os.path.join(self.dataset_path, self.data_list[index].replace('GT', 'SAR'))
        pre_rgb_path = os.path.join(self.dataset_path.replace('UrbanSARFloods_v1_cut_8', 'VHR/pre_event_cut_8'), self.data_list[index])
        prior_data_path = os.path.join(self.dataset_path.replace('UrbanSARFloods_v1_cut_8', 'damagePlain_cut_8'), self.data_list[index])
        loc_label_path = os.path.join(self.dataset_path.replace('UrbanSARFloods_v1_cut_8', 'USBuildingFootprints_10240_cut_8'), self.data_list[index])
        clf_label_path = os.path.join(self.dataset_path.replace('UrbanSARFloods_v1_cut_8', 'PDE_10240_cut_8'), self.data_list[index])
        map_label_path = os.path.join(self.dataset_path, self.data_list[index])
        if os.path.exists(img_path):
            img = self.loader(img_path)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        else:
            img = np.ones((size, size, 8), dtype=np.uint8) * 255 # Placeholder for missing images
        pre_img = img[:, :, [0, 1, 4, 5]]
        post_img = img[:, :, [2, 3, 6, 7]]
        if os.path.exists(pre_rgb_path):
            pre_rgb = self.loader(pre_rgb_path)
        else:
            pre_rgb = np.ones_like(pre_img[:,:,:3]) * 255
        if os.path.exists(prior_data_path):      
            prior_data = self.loader(prior_data_path)
        else:
            prior_data = np.ones_like(pre_img[:,:,0]) * 255
            prior_data = prior_data.reshape((size, size, 1))
        if os.path.exists(loc_label_path):      
            loc_label = self.loader(loc_label_path)[:,:,0]
        else:
            loc_label = np.ones_like(pre_img[:,:,0]) * 255
        if os.path.exists(clf_label_path):
            clf_label = self.loader(clf_label_path)[:,:,0]
        else:
            clf_label = np.ones_like(pre_img[:,:,0]) * 255
        if os.path.exists(map_label_path):
            map_label = self.loader(map_label_path)[:,:,0]
            map_label = cv2.resize(map_label, (size, size), interpolation=cv2.INTER_NEAREST)
        else:
            map_label = np.ones_like(pre_img[:,:,0]) * 255

        prior_data[np.isnan(prior_data)] = 255
        prior_data[np.isnan(pre_img).sum(axis=2) != 0] = 255
        prior_data[np.isnan(post_img).sum(axis=2) != 0] = 255
        loc_label[np.isnan(loc_label)] = 255
        loc_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
        clf_label[clf_label < 13.66] = 1
        clf_label[(clf_label >= 13.66) & (clf_label < 38.97)] = 2
        clf_label[(clf_label >= 38.97) & (clf_label < 254)] = 3
        clf_label[clf_label == 254] = 0
        clf_label[np.isnan(clf_label)] = 255
        clf_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
        clf_label[np.isnan(post_img).sum(axis=2) != 0] = 255
        map_label[np.isnan(map_label)] = 255
        map_label[np.isnan(pre_img).sum(axis=2) != 0] = 255
        map_label[np.isnan(post_img).sum(axis=2) != 0] = 255
        pre_img[np.isnan(pre_img)] = 255
        post_img[np.isnan(post_img)] = 255
        pre_rgb[np.isnan(pre_rgb)] = 255

        if 'train' in self.data_pro_type:
            pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label = self.__transforms(True, pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label)

        else:
            pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label = self.__transforms(False, pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label)
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)
            map_label = np.asarray(map_label)

        data_idx = self.data_list[index]
        return pre_img, post_img, pre_rgb, prior_data, loc_label, clf_label, map_label, data_idx

    def __len__(self):
        return len(self.data_list)


def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'UrbanSARFloods_Base' in args.dataset:
        dataset = DamageAssessmentDataset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_epochs, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader

    elif args.dataset == 'UrbanSARFloods_Fusion':
        dataset = DamageAssessmentFusionDataset(args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_epochs, args.type)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="UrbanSARFloods_Base DataLoader Test")
    parser.add_argument('--dataset', type=str, default='UrbanSARFloods_Base')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--dataset_path', type=str, default='../data/UrbanSARFloods_v1')
    parser.add_argument('--data_list_path', type=str, default='../data/UrbanSARFloods_v1/splits/Train_dataset.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--data_name_list', type=list)

    args = parser.parse_args()

    with open(args.data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list
    train_data_loader = make_data_loader(args)
    for i, data in enumerate(train_data_loader):
        pre_img, post_img, labels, _ = data
        pre_data, post_data = Variable(pre_img), Variable(post_img)
        labels = Variable(labels)
        print(i, "inputs", pre_data.data.size(), "labels", labels.data.size())
