import argparse
import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from load_data import RegistrationData, ModelNet40Data

from net import MFINet
from module import Encoder

from losses import ChamferDistanceLoss
from losses import computeLoss

from metrics import compute_metrics, summary_metrics, print_metrics
from utils.process import batch_quat2mat

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def test_one_epoch(device, model, test_loader):
    model.eval()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    test_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt = data

        template = template.cuda()
        source = source.cuda()
        igt = igt.cuda()

        output = model(template, source)
        loss_val = ChamferDistanceLoss()(template, output['transformed_source'])
        #loss_val = computeLoss()(output['pose_7d'],igt[:,0,:],output['tgt_feat'],output['src_feat'])

        test_loss += loss_val.item()
        count += 1

        igtR = batch_quat2mat(igt[:, 0, :4]).cuda()
        igtt = igt[:, 0, 4:7].cuda()
        R = output['est_R']  # 3维
        t = output['est_t'][:, 0, :]  # 2维

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, igtR, igtt)
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    # print_metrics('mfi', 0, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    test_loss = float(test_loss) / count

    results = {
        'test_loss': test_loss,
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }

    return results


def test(args, model, test_loader, textio):
    test_result = test_one_epoch(args.device, model, test_loader)
    textio.cprint('Validation Loss: %f ' % (test_result['r_isotropic']))

def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    train_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt = data

        # template = template.to(device)
        # source = source.to(device)
        # igt = igt.to(device)
        template = template.cuda()
        source = source.cuda()
        igt = igt.cuda()

        output = model(template, source)
        loss_val = ChamferDistanceLoss()(template, output['transformed_source'])
        #loss_val = computeLoss()(output['pose_7d'], igt[:, 0, :],output['tgt_feat'],output['src_feat'])
        # print(loss_val.item())

        # forward + backward + optimize
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        train_loss += loss_val.item()
        count += 1

        igtR = batch_quat2mat(igt[:, 0, :4]).cuda()
        igtt = igt[:, 0, 4:7].cuda()
        R = output['est_R']  # 3维
        t = output['est_t'][:, 0, :]  # 2维

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, igtR, igtt)
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    #print_metrics('mfi', 0, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    train_loss = float(train_loss) / count

    results = {
        'train_loss': train_loss,
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=0.0001)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_test_loss = np.inf
    # best_test_r_isotropic = np.inf
    # best_train_r_isotropic = np.inf
    best_test_r_mse = np.inf
    best_train_r_mse = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        train_result = train_one_epoch(args.device, model, train_loader, optimizer)
        test_result = test_one_epoch(args.device, model, test_loader)

        if train_result['r_mse'] < best_train_r_mse:
            best_train_r_mse= train_result['r_mse']
            torch.save(model.state_dict(), 'checkpoints/%s/models/best_train.t7' % (args.exp_name))

        if test_result['r_mse'] < best_test_r_mse:
            best_test_r_mse= test_result['r_mse']
            best_test_loss = test_result['test_loss']
            # snap = {'epoch': epoch + 1,
            #         'model': model.state_dict(),
            #         'min_loss': best_test_loss,
            #         'optimizer': optimizer.state_dict(), }
            #torch.save(snap, 'checkpoints/%s/models/best_model_snap.t7' % (args.exp_name))
            torch.save(model.state_dict(), 'checkpoints/%s/models/best_test.t7' % (args.exp_name))
            #torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/best_ptnet_model.t7' % (args.exp_name))

        #torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % (args.exp_name))
        torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % (args.exp_name))
        #torch.save(model.feature_model.state_dict(), 'checkpoints/%s/models/ptnet_model.t7' % (args.exp_name))

        boardio.add_scalar('Train Loss', train_result['train_loss'], epoch + 1)
        boardio.add_scalar('Test Loss', test_result['test_loss'], epoch + 1)
        boardio.add_scalar('Best Test Loss', best_test_loss, epoch + 1)
        boardio.add_scalar('Best Test r_mse', best_test_r_mse, epoch + 1)
        boardio.add_scalar('Best Train r_mse', best_train_r_mse, epoch + 1)

        boardio.add_scalar('Train r_mse', train_result['r_mse'], epoch + 1)
        boardio.add_scalar('Train r_mae', train_result['r_mae'], epoch + 1)
        boardio.add_scalar('Train t_mse', train_result['t_mse'], epoch + 1)
        boardio.add_scalar('Train t_mae', train_result['t_mae'], epoch + 1)
        boardio.add_scalar('Train r_isotropic', train_result['r_isotropic'], epoch + 1)
        boardio.add_scalar('Train t_isotropic', train_result['t_isotropic'], epoch + 1)

        boardio.add_scalar('Test r_mse', test_result['r_mse'], epoch + 1)
        boardio.add_scalar('Test r_mae', test_result['r_mae'], epoch + 1)
        boardio.add_scalar('Test t_mse', test_result['t_mse'], epoch + 1)
        boardio.add_scalar('Test t_mae', test_result['t_mae'], epoch + 1)
        boardio.add_scalar('Test r_isotropic', test_result['r_isotropic'], epoch + 1)
        boardio.add_scalar('Test t_isotropic', test_result['t_isotropic'], epoch + 1)

        textio.cprint('\nEPOCH:%d, (Traininig) --- Loss:%f, Error(r):%f, r_mse:%f, r_mae:%f, Error(t):%f, t_mse:%f, t_mae:%f'
                      %(epoch+1, train_result['train_loss'],train_result['r_isotropic'], train_result['r_mse'], train_result['r_mae'],train_result['t_isotropic'], train_result['t_mse'], train_result['t_mae']))
        textio.cprint('\nEPOCH:%d, (Testinig) --- Loss:%f, Error(r):%f, r_mse:%f, r_mae:%f, Error(t):%f, t_mse:%f, t_mae:%f'
                      %(epoch+1, test_result['test_loss'],test_result['r_isotropic'], test_result['r_mse'], test_result['r_mae'], test_result['t_isotropic'], test_result['t_mse'], test_result['t_mae']))
        textio.cprint('\nEPOCH:%d, (Trained) Best Error(r):%f --- (Tested) Best Loss:%f, Best r_mse:%f'%(epoch+1, best_train_r_mse, best_test_loss, best_test_r_mse ))

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='MFInet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_path', type=str, default='ModelNet40',
                        metavar='PATH', help='path to the input dataset')  # like '/path/to/ModelNet40'
    parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-j', '--workers', default=2, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=100, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')

    parser.add_argument('--device', default='cuda:1', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args


def main():
    args = options()

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
    _init_(args)

    textio = IOStream('checkpoints/' + args.exp_name + '/run_new.log')
    textio.cprint(str(args))

    trainset = RegistrationData(ModelNet40Data(train=True))
    testset = RegistrationData(ModelNet40Data(train=False))

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers)
    test_loader = DataLoader(testset, batch_size=4, shuffle=False, drop_last=False,
                             num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)
    torch.cuda.set_device(0)

    pn = Encoder(8,4)
    model = MFINet(pn)
    #model = model.to(args.device)
    model = model.cuda()

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    #model.to(args.device)
    model = model.cuda()

    if args.eval:
        test(args, model, test_loader, textio)
    else:
        train(args, model, train_loader, test_loader, boardio, textio, checkpoint)

if __name__ == '__main__':
    main()
