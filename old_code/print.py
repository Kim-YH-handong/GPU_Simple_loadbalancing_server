import argparse

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

if __name__ == "__main__":
    # print("Hello")
    print(args.volume_path)