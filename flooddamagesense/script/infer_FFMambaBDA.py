import sys
sys.path.append('/home/grads/y/yuhsuanho/Flood-DamageSense/')

import argparse
import os
import time
import numpy as np
import imageio
import rasterio

from flooddamagesense.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from flooddamagesense.datasets.make_data_loader import make_data_loader, DamageAssessmentFusionDataset
from flooddamagesense.utils_func.metrics import Evaluator
from flooddamagesense.models.FFMambaBDA import FFMambaBDA


def get_model(args):
    if args.model_type == 'FFMambaBDA_Tiny':
        args.pretrained_weight_path = '../pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth'
        args.cfg = './configs/vssm1/vssm_tiny_224_0229flex.yaml'
    elif args.model_type == 'FFMambaBDA_Small':
        args.pretrained_weight_path = '../pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth'
        args.cfg = './configs/vssm1/vssm_small_224.yaml'
    elif args.model_type == 'FFMambaBDA_Base':
        args.pretrained_weight_path = '../pretrained_weight/vssm_base_0229_ckpt_epoch_237.pth'
        args.cfg = './configs/vssm1/vssm_base_224.yaml'
    else:
        raise NotImplementedError
    
    config = get_config(args)
    model = FFMambaBDA(
            output_building=2, output_damage=4, output_map=3, 
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            )
    return model

class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.evaluator_map = Evaluator(num_class=3)

        self.deep_model = get_model(args)
        self.deep_model = self.deep_model.cuda()
        self.lr = args.learning_rate

        self.building_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'building_localization_map')
        self.damage_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'damage_classification_map')
        self.flood_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'flooding_map')

        if not os.path.exists(self.building_map_saved_path):
            os.makedirs(self.building_map_saved_path)
            os.makedirs(os.path.join(self.building_map_saved_path, '01_NF/GT'))
            os.makedirs(os.path.join(self.building_map_saved_path, '02_FO/GT'))
            os.makedirs(os.path.join(self.building_map_saved_path, '03_FU/GT'))
        if not os.path.exists(self.damage_map_saved_path):
            os.makedirs(self.damage_map_saved_path)
            os.makedirs(os.path.join(self.damage_map_saved_path, '01_NF/GT'))
            os.makedirs(os.path.join(self.damage_map_saved_path, '02_FO/GT'))
            os.makedirs(os.path.join(self.damage_map_saved_path, '03_FU/GT'))
        if not os.path.exists(self.flood_map_saved_path):
            os.makedirs(self.flood_map_saved_path)
            os.makedirs(os.path.join(self.flood_map_saved_path, '01_NF/GT'))
            os.makedirs(os.path.join(self.flood_map_saved_path, '02_FO/GT'))
            os.makedirs(os.path.join(self.flood_map_saved_path, '03_FU/GT'))


        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()


    def infer(self):
        torch.cuda.empty_cache()
        dataset = DamageAssessmentFusionDataset(self.args.test_dataset_path, self.args.test_data_name_list, self.args.crop_size, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_map.reset()          
        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for _, data in enumerate(tqdm(val_data_loader)):
                pre_change_imgs, post_change_imgs, pre_rgb_imgs, prior_data, labels_loc, labels_clf, labels_map, names = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                pre_rgb_imgs = pre_rgb_imgs.cuda()
                prior_data = prior_data.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()
                labels_map = labels_map.cuda().long()

                output_loc, output_clf, output_map = self.deep_model(pre_change_imgs, post_change_imgs, pre_rgb_imgs, prior_data)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                output_map = output_map.data.cpu().numpy()
                output_map = np.argmax(output_map, axis=1)
                labels_map = labels_map.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)

                output_clf_eval = output_clf[labels_loc > 0]
                labels_clf_eval = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf_eval, output_clf_eval)

                self.evaluator_map.add_batch(labels_map, output_map)
    
                image_name = names[0][:-4] + '.png'

                output_loc = np.squeeze(output_loc)

                output_clf[labels_loc == 0] = 255
                output_clf = np.squeeze(output_clf)

                output_map = np.squeeze(output_map)

                imageio.imwrite(os.path.join(self.building_map_saved_path, image_name), output_loc.astype(np.uint8))
                imageio.imwrite(os.path.join(self.damage_map_saved_path, image_name), output_clf.astype(np.uint8))
                imageio.imwrite(os.path.join(self.flood_map_saved_path, image_name), output_map.astype(np.uint8))

                gt_path = os.path.join(self.args.test_dataset_path.replace('UrbanSARFloods_v1_cut_8', 'PDE_10240_cut_8'), names[0][:-4] + '.tif')

                with rasterio.open(gt_path, 'r') as src:
                    img = src.read()
                    with rasterio.open(
                        os.path.join(self.damage_map_saved_path, names[0][:-4] + '.tif'),
                        'w',
                        driver='GTiff',
                        height=img.shape[1],
                        width=img.shape[2],
                        count=1,
                        dtype=rasterio.uint8,
                        crs=src.crs,
                        transform=src.transform,
                    ) as dst:
                        dst.write(output_clf, 1)
        
        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score[1:]) / np.sum(1.0 / damage_f1_score[1:])
        map_f1_score = self.evaluator_map.Damage_F1_score()
        map_harmonic_mean_f1 = len(map_f1_score[1:]) / np.sum(1.0 / map_f1_score[1:])
        oaf1 = 0.1 * loc_f1_score + 0.7 * harmonic_mean_f1 + 0.2 * map_harmonic_mean_f1

        print(f'locF1 is {loc_f1_score}, clfF1 is {harmonic_mean_f1}, mapF1 is {map_harmonic_mean_f1}, oaF1 is {oaf1}, '
            f'sub class F1 score is {damage_f1_score}, sub class f1_score for flood mapping is {map_f1_score}')

def main():
    parser = argparse.ArgumentParser(description="Inference on UrbanSARFloods_Fusion dataset")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='UrbanSARFloods_Fusion')
    parser.add_argument('--type', type=str, default='test')
    parser.add_argument('--test_dataset_path', type=str, default='../data/UrbanSARFloods_v1_cut_8')
    parser.add_argument('--test_data_list_path', type=str, default='../data/UrbanSARFloods_v1/splits/Test_dataset.txt')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--model_type', type=str, default='FFMambaBDA_Base')
    parser.add_argument('--result_saved_path', type=str, default='../results')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip()[:-4]+'_{}.tif'.format(i) for data_name in f for i in range(64) if os.path.exists(os.path.join(args.test_dataset_path.replace('UrbanSARFloods_v1_cut_8', 'PDE_10240_cut_8'), data_name.strip()[:-4]+'_{}.tif'.format(i))) and '20170830_Houston' in data_name.strip()]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.infer()


if __name__ == "__main__":
    main()
