import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_vfss import VFSS_dataset, RandomGenerator_test
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='/home/harim/vfss_dt/Data/Dataset_Ver02', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='VFSS', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=250,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224,
                    help='input patch size of network input')
parser.add_argument('--n_skip', type=int, default=3,
                    help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--test_case', type=str, default='test1', help='test case')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = VFSS_dataset(base_dir=args.volume_path,
                           transform=transforms.Compose(
                               [RandomGenerator_test(output_size=[args.img_size, args.img_size])]))
    testloader = DataLoader(db_test, batch_size=1,
                            shuffle=False, num_workers=1)
    model.eval()
    name = ['bolus', 'cervical spine', 'hyoid bone',
            'mandible', 'soft tissue', 'vocal folds']
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                                 test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
    return "Testing Finished!"


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'VFSS': {
            'Dataset': VFSS_dataset,
            'volume_path': '/home/younghun/VFSS/Data_0523Split[Undersampling]/[Dataset2]CLAHE+old+new_0316',
            'num_classes': 6,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    dataset_config[dataset_name]['volume_path'] = args.volume_path
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(
            args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))

    # Load Model
    net = ViT_seg(config_vit, img_size=args.img_size,
                  num_classes=config_vit.n_classes).cuda()
    snapshot = '/home/younghun/IoT/server/prediction_model.pth'
    net.load_state_dict(torch.load(snapshot))

    # Save Path
    test_save_path = os.path.join(args.volume_path, 'result_img')
    os.makedirs(test_save_path, exist_ok=True)

    print(inference(args, net, test_save_path))
