import sys
sys.path.append('/home/harim/vfss_dt/Data_Phase2_Experiments/TEST2-7')
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from test2_7_iot import inference
import argparse
import logging
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import openpyxl as op
from openpyxl import Workbook
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from datasets.dataset_vfss import VFSS_dataset, RandomGenerator_test
from torchvision import transforms



GPU_NUM={False,False,False,False,False,False,False,False}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())
# print(torch.cuda.get_device_name(device))


def get_available_gpu():
    gpu_list = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_list.append((i, gpu_name))
        print('gpu_list')
        print(gpu_list)
    return gpu_list


available_gpus = get_available_gpu()
print("사용 가능한 GPU:")
for gpu in available_gpus:
    print(f"GPU 번호: {gpu[0]}, 이름: {gpu[1]}")

# 로드 밸런싱을 위한 GPU 할당

def get_next_available_gpu():
    next_gpu_idx = torch.cuda.current_device() + 1
    if next_gpu_idx >= len(available_gpus):
        next_gpu_idx = 0
    next_gpu_idx = available_gpus[next_gpu_idx][0]
    print('get_next_available_gpu: '+str(next_gpu_idx))
    return next_gpu_idx

gpu_num = get_next_available_gpu()
print('>>> gpu_num: '+str(gpu_num))
# CUDA_VISIBLE_DEVICES=gpu_num 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
device = torch.device(
f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
print('==>>> Current cuda device ', torch.cuda.current_device())  # check
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    default='VFSS', help='experiment_name')
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
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    snapshot_path = "/home/harim/vfss_dt/Data_Phase2_Experiments/TEST2-7/Result/" + args.test_case

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)

    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(
            args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    
    # # Additional Infos
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(gpu_num))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(gpu_num)/1024**3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(gpu_num)/1024**3, 1), 'GB')

    net = ViT_seg(config_vit, img_size=args.img_size,
                  num_classes=config_vit.n_classes).cuda(gpu_num)

    snapshot = '/home/younghun/VFSS/Data_Phase2_Experiments/TEST2-7/Result/test2-7_newdb0523/test2-7_newdb0523_bestmodel_epoch:86iternum:7830.pth'
    net.load_state_dict(torch.load(snapshot))
    snapshot_name = snapshot_path.split('/')[-1]

    log_testfile = '/home/harim/vfss_dt/Data_Phase2_Experiments/TEST2-7/Result/'+args.test_case + \
        '/logtest_' + args.test_case + \
        '_bestmodel_epoch:86iternum:7830.txt'
    logging.basicConfig(filename=log_testfile, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    test_save_path = './Result/'+args.test_case + '/prediction'
    grad_save_path = './Result/'+args.test_case + \
        '/gradCAM_net.transformer.embeddings.hybrid_model.body.block3.unit9.relu'
    os.makedirs(test_save_path, exist_ok=True)
    os.makedirs(grad_save_path, exist_ok=True)

    inference(args, net, test_save_path, grad_save_path, gpu_number=gpu_num)
