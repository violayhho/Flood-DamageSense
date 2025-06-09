import sys
sys.path.append('/home/grads/y/yuhsuanho/Flood-DamageSense/')

import argparse
import os
import time

import numpy as np

from flooddamagesense.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from flooddamagesense.datasets.make_data_loader import make_data_loader, DamageAssessmentFusionDataset
from flooddamagesense.utils_func.metrics import Evaluator
from flooddamagesense.models.FFMambaBDA import FFMambaBDA

import flooddamagesense.utils_func.lovasz_loss as L


def build_write_logfile(model_type):
    stamp = str(time.time())
    def write_logfile(message_to_print):
        print(message_to_print)
        log_file='log_files/{}_output_{}.txt'.format(model_type, stamp)
        with open(log_file, 'a') as of:
            of.write(message_to_print + '\n')
    return write_logfile, stamp

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
        self.write_logfile, stamp = build_write_logfile(args.model_type)

        self.train_data_loader = make_data_loader(args)

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=4)
        self.evaluator_map = Evaluator(num_class=3)

        self.deep_model = get_model(args) 
        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + stamp)
        self.lr = args.learning_rate

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

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

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        self.write_logfile('Start training')
        accumulation_steps = 8
        best_damage_f1 = 0.0
        best_epoch = 0
        best_scores = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        data_num = len(self.args.train_data_name_list)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, pre_rgb_imgs, prior_data, labels_loc, labels_clf, labels_map, _ = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            pre_rgb_imgs = pre_rgb_imgs.cuda()
            prior_data = prior_data.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()
            labels_map = labels_map.cuda().long()
            
            output_loc, output_clf, output_map = self.deep_model(pre_change_imgs, post_change_imgs, pre_rgb_imgs, prior_data)

            final_loss = torch.tensor(0., requires_grad=True).cuda()
            if not torch.all(labels_loc == 255):
                ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
                lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)
                final_loss += self.args.ce_loss_loc_weight * ce_loss_loc + self.args.lovasz_loss_loc_weight * lovasz_loss_loc
            else:
                ce_loss_loc = 0
                lovasz_loss_loc = 0

            clf_weights = [107972/60124/2, 107972/60124, 107972/36068, 107972/11780]
            if not torch.all(labels_clf == 255):
                ce_loss_clf = F.cross_entropy(output_clf, labels_clf, ignore_index=255, weight=torch.tensor(clf_weights).cuda())
                lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255, weight=clf_weights)
                final_loss += self.args.ce_loss_clf_weight * ce_loss_clf + self.args.lovasz_loss_clf_weight * lovasz_loss_clf
            else:
                ce_loss_clf = 0
                lovasz_loss_clf = 0

            map_weights = [7442/4770, 7442/2981, 7442/690]
            if not torch.all(labels_map == 255):
                ce_loss_map = F.cross_entropy(output_map, labels_map, ignore_index=255, weight=torch.tensor(map_weights).cuda())
                lovasz_loss_map = L.lovasz_softmax(F.softmax(output_map, dim=1), labels_map, ignore=255, weight=map_weights)
                final_loss += self.args.ce_loss_map_weight * ce_loss_map + self.args.lovasz_loss_map_weight * lovasz_loss_map
            else:
                ce_loss_map = 0
                lovasz_loss_map = 0

            final_loss /= accumulation_steps
            final_loss.backward()

            #self.optim.step()
                    
            if ((itera + 1) % accumulation_steps) == 0:
                self.optim.step()
                self.optim.zero_grad()

            if (itera + 1) % 10 == 0:
                self.write_logfile(f'iter is {itera + 1}, localization loss is {ce_loss_loc + lovasz_loss_loc}, classification loss is {ce_loss_clf + lovasz_loss_clf}, flood mapping loss is {ce_loss_map + lovasz_loss_map}')
            if (itera * self.args.batch_size) % data_num >= (data_num - self.args.batch_size):
                epoch = (itera + 1) * self.args.batch_size // data_num
                epoch += self.args.start_epoch
                torch.save(self.deep_model.state_dict(),
                            os.path.join(self.model_save_path, f'{epoch}_model.pth'))
                self.write_logfile(f'epoch is {epoch}, localization loss is {ce_loss_loc + lovasz_loss_loc}, classification loss is {ce_loss_clf + lovasz_loss_clf}, flood mapping loss is {ce_loss_map + lovasz_loss_map}')
                self.deep_model.eval()
                loc_f1_score, harmonic_mean_f1, map_harmonic_mean_f1, damage_f1_score, map_f1_score, oaf1 = self.validation()
                
                if harmonic_mean_f1 > best_damage_f1:
                    best_damage_f1 = harmonic_mean_f1
                    best_epoch = epoch
                    best_scores = [loc_f1_score, harmonic_mean_f1, map_harmonic_mean_f1, oaf1, damage_f1_score, map_f1_score]
                self.deep_model.train()

        self.write_logfile(f'Best epoch is {best_epoch}')
        self.write_logfile(f'Best scores: locF1 is {best_scores[0]}, clfF1 is {best_scores[1]}, mapF1 is {best_scores[2]}, '
              f'oaF1 is {best_scores[3]}, sub class F1 score is {best_scores[4]}, sub class f1_score for flood mapping is {best_scores[5]}')


    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        self.evaluator_map.reset()
        dataset = DamageAssessmentFusionDataset(self.args.test_dataset_path, self.args.test_data_name_list, self.args.crop_size, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, pre_rgb_imgs, prior_data, labels_loc, labels_clf, labels_map, _ = data

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
                
                output_clf = output_clf[labels_loc > 0]
                labels_clf = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf, output_clf)

                self.evaluator_map.add_batch(labels_map, output_map)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.evaluator_clf.Damage_F1_score()
        harmonic_mean_f1 = len(damage_f1_score[1:]) / np.sum(1.0 / damage_f1_score[1:])
        map_f1_score = self.evaluator_map.Damage_F1_score()
        map_harmonic_mean_f1 = len(map_f1_score[1:]) / np.sum(1.0 / map_f1_score[1:])
        oaf1 = 0.1 * loc_f1_score + 0.7 * harmonic_mean_f1 + 0.2 * map_harmonic_mean_f1

        self.write_logfile(f'locF1 is {loc_f1_score}, clfF1 is {harmonic_mean_f1}, mapF1 is {map_harmonic_mean_f1}, oaF1 is {oaf1}, '
              f'sub class F1 score is {damage_f1_score}, sub class f1_score for flood mapping is {map_f1_score}')
        return loc_f1_score, harmonic_mean_f1, map_harmonic_mean_f1, oaf1, damage_f1_score, map_f1_score


def main():
    parser = argparse.ArgumentParser(description="Training on UrbanSARFloods_Fusion dataset")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='UrbanSARFloods_Fusion')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='../data/UrbanSARFloods_v1_cut_8')
    parser.add_argument('--train_data_list_path', type=str, default='../data/UrbanSARFloods_v1/splits/Train_dataset.txt')
    parser.add_argument('--test_dataset_path', type=str, default='../data/UrbanSARFloods_v1_cut_8')
    parser.add_argument('--test_data_list_path', type=str, default='../data/UrbanSARFloods_v1/splits/Valid_dataset.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--model_type', type=str, default='FFMambaBDA_Base')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    parser.add_argument('--ce_loss_loc_weight', type=float, default=1)
    parser.add_argument('--ce_loss_clf_weight', type=float, default=1)
    parser.add_argument('--ce_loss_map_weight', type=float, default=1)
    parser.add_argument('--lovasz_loss_loc_weight', type=float, default=0.5)
    parser.add_argument('--lovasz_loss_clf_weight', type=float, default=0.75)
    parser.add_argument('--lovasz_loss_map_weight', type=float, default=0.5)
            
    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        data_name_list = [data_name.strip()[:-4]+'_{}.tif'.format(i) for data_name in f for i in range(64) if os.path.exists(os.path.join(args.train_dataset_path, data_name.strip()[:-4]+'_{}.tif'.format(i))) and '20170830_Houston' in data_name.strip()]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        test_data_name_list = [data_name.strip()[:-4]+'_{}.tif'.format(i) for data_name in f for i in range(64) if os.path.exists(os.path.join(args.test_dataset_path.replace('UrbanSARFloods_v1_cut_8', 'PDE_10240_cut_8'), data_name.strip()[:-4]+'_{}.tif'.format(i))) and '20170830_Houston' in data_name.strip()]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
