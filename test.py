import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
import os
import numpy as np
from load_data import RegistrationData, ModelNet40Data
from losses import ChamferDistanceLoss
from load_stanf import CustomData

from net import MFINet
from module import Encoder

# from MFINet import MFINet
# from LAGNet import Encoder

from metrics import compute_metrics, summary_metrics, print_metrics
from utils.process import batch_quat2mat

def display_open3d(template, source, transformed_source):
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    transformed_source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template)
    source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
    transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
    template_.paint_uniform_color([1, 0, 0])
    source_.paint_uniform_color([0, 1, 0])
    transformed_source_.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([template_, source_, transformed_source_])

def test_one_epoch(device, model, test_loader):
    model.eval()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []

    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt = data

        # template = template.to(device)
        # source = source.to(device)
        template = template.cuda()
        source = source.cuda()
        output = model(template, source)

        # igtR = batch_quat2mat(igt[:,0,:4]).to(device)
        # igtt = igt[:,0, 4:7].to(device)
        igtR = batch_quat2mat(igt[:, 0, :4]).cuda()
        igtt = igt[:,0, 4:7].cuda()
        R = output['est_R'] #3维
        t = output['est_t'][:,0,:] #2维

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, igtR, igtt)
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

        #display_open3d(template.detach().cpu().numpy()[0], source.detach().cpu().numpy()[0], output['transformed_source'].detach().cpu().numpy()[0])

    mr_mse, mr_mae, mt_mse, mt_mae, mr_isotropic, mt_isotropic = summary_metrics(r_mse, r_mae, t_mse, t_mae,
                                                                                 r_isotropic, t_isotropic)
    print_metrics('mfi', 0, mr_mse, mr_mae, mt_mse, mt_mae, mr_isotropic, mt_isotropic)

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')

    parser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('-j', '--workers', default=2, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 32)')

    parser.add_argument('--pretrainedPCR', default='best_test_v19.t7', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')

    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args

def main():
    args = options()

    testset = RegistrationData(ModelNet40Data(),noise_source=True)
    #testset = CustomData(1536)
    test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    #args.device = torch.device(args.device)
    torch.cuda.set_device(0)

    # Create PointNet Model.
    pn = Encoder(16, 4)
    model = MFINet(pn)

    if args.pretrainedPCR:
        assert os.path.isfile(args.pretrainedPCR)
        model.load_state_dict(torch.load(args.pretrainedPCR, map_location='cpu'))
    #model.to(args.device)
    model = model.cuda()

    test_one_epoch(args.device, model, test_loader)

if __name__ == '__main__':
    main()