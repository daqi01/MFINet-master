import torch
import torch.nn as nn
import torch.nn.functional as F

def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2, dim=0, keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices

def knn(x, k):  # x: data(B, 3, N)  k: neighbors_num
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)    # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)     # 对第1维求平方和, keepdim:求和之后这个dim的元素个数为１，所以要被去掉，如果要保留这个维度，则应当keepdim=True`
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_neighbors(data, k=20):
    # xyz = data[:, :3, :]    # (B, 3, N)
    xyz = data.view(*data.size()[:3])
    idx = knn(xyz, k=k)  # (batch_size, num_points, k) 即: (B, N, n): 里面存的是N个点的n个邻居的下标
    batch_size, num_points, _ = idx.size()
    # device = torch.device('cuda')

    # idx_base: [B, 1, 1]
    idx_base = torch.arange(0, batch_size).to(xyz.device).view(-1, 1, 1) * num_points   # arange不包含batch_size
    nbrs = torch.tensor([]).to(xyz.device)

    idx = idx + idx_base    # 每个点n近邻的下标    (B, N, n)
    idx = idx.view(-1)  # idx: 0 ~ (batch_size * num_points -1)

    _, num_dims, _ = xyz.size()    # num_dims = 3

    xyz = xyz.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)

    # gxyz
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    #  batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_gxyz = xyz.view(batch_size * num_points, -1)[idx, :]   # neighbor_gxyz.shape = (B*N*n, 3)
    neighbor_gxyz = neighbor_gxyz.view(batch_size, num_points, k, num_dims)     # (B, N, n, 3)
    # if 'gxyz' in feature_name:
    #     net_input = torch.cat((net_input, neighbor_gxyz), dim=3)

    # # xyz
    # xyz = xyz.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # net_input = torch.cat((net_input, xyz), dim=3)

    nbrs = torch.cat((nbrs, neighbor_gxyz), dim=3)
    nbrs = nbrs.permute(0, 3, 1, 2).contiguous()

    return nbrs, idx    # (B, 3, N, n)

class FusionMoudel(nn.Module):
    def __init__(self, channel, reduction=4):   # channel = 1024
        super(FusionMoudel, self).__init__()
        self.fcn_1 = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1), nn.BatchNorm1d(channel // reduction), nn.ReLU(), #1024,256
            nn.Conv1d(channel // reduction, 1, 1), nn.BatchNorm1d(1)   #256 1
        )
        self.fcn_2 = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1), nn.BatchNorm1d(channel // reduction), nn.ReLU(),
            nn.Conv1d(channel // reduction, 1, 1), nn.BatchNorm1d(1)
        )
        # self.fcn_3 = nn.Sequential(
        #     nn.Conv1d(channel, channel // reduction, 1), nn.BatchNorm1d(channel // reduction), nn.ReLU(),
        #     nn.Conv1d(channel // reduction, 1, 1), nn.BatchNorm1d(1)
        # )

    # 局部全局融合(或多个局部融合)
    def forward(self, feature1, feature2): #(B C N)
        feature1 = feature1.unsqueeze(dim=1)
        feature2 = feature2.unsqueeze(dim=1)
        features = torch.cat((feature1, feature2), dim=1) # (B, 2, C, N)
        feature_U = torch.sum(features, dim=1)  # (B, C, N)

        # a + b = 1
        a = self.fcn_1(feature_U)   # (B, 1, N)
        b = self.fcn_2(feature_U)   # (B, 1, N)
        matrix = torch.cat((a, b), dim=1)   # (B, 2, N)
        matrix = F.softmax(matrix, dim=1)   # g -> a; f -> 1-a (B, 2, N)
        matrix = matrix.unsqueeze(dim=2)    # (B, 2, 1, N)
        features = (matrix * features).sum(dim=1)  # (B, C, N): a * g + (1 - a) * f

        return features

class FFNet(nn.Module):
    def __init__(self):
        super(FFNet, self).__init__()

        self.pa_layer = FusionMoudel(channel=1024, reduction=4)

        self.conv1d_6 = nn.Conv1d(1024, 1024, 1)
        self.bn1d_6 = nn.BatchNorm1d(1024)
        # self.conv1d_7 = nn.Conv1d(1024, 1024, 1)
        # self.bn1d_7 = nn.BatchNorm1d(1024)

    def forward(self, lf1_m, lf2_m):
        fuse = self.pa_layer(lf1_m, lf2_m)  # (B, 1024, N)

        batch_size, num_dims, N = fuse.size()

        pointcloud_features = F.relu(self.bn1d_6(self.conv1d_6(fuse)), inplace=True)
        # pointcloud_features = F.relu(self.bn1d_7(self.conv1d_7(pointcloud_features)), inplace=True)

        rand_mask = torch.rand((batch_size, 1, N)) > 0.3
        rand_mask = rand_mask.cuda()
        pointcloud_features = pointcloud_features * rand_mask

        return pointcloud_features  # (B, 1024, N)

class Encoder(nn.Module):
    def __init__(self, nbrs_num1=16, nbrs_num2=4):
        super(Encoder, self).__init__()

        self.nbrs_num1 = nbrs_num1
        self.nbrs_num2 = nbrs_num2

        self.conv2d_1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.conv2d_2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv2d_4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        #self.conv2d_5 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
        self.bn2d_1 = nn.BatchNorm2d(64)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.bn2d_3 = nn.BatchNorm2d(128)
        self.bn2d_4 = nn.BatchNorm2d(256)
        #self.bn2d_5 = nn.BatchNorm2d(512)

        self.conv1d_6 = nn.Conv1d(512, 512, 1)
        self.bn1d_6 = nn.BatchNorm1d(512)
        self.conv1d_7 = nn.Conv1d(512, 1024, 1)
        self.bn1d_7 = nn.BatchNorm1d(1024)

    def forward(self, pointcloud):
        pointcloud = pointcloud.permute(0, 2, 1).contiguous()   # (32, 3, 1024) 即:(B, 3, N)
        batch_size, num_dims, N = pointcloud.size()

        #随机产生0.3的dropout
        rand_mask_f1 = torch.rand((batch_size, 1, N, self.nbrs_num1)) > 0.3
        rand_mask_f1 = rand_mask_f1.cuda()
        rand_mask_f2 = torch.rand((batch_size, 1, N, self.nbrs_num2)) > 0.3
        rand_mask_f2 = rand_mask_f2.cuda()

        # 仅局部多尺度融合
        lf1, idx_lf1 = get_neighbors(pointcloud, k=self.nbrs_num1)  # (B, 3, N, n1)
        lf2 = lf1[:, :, :, :self.nbrs_num2]

        #第一个分支
        lf1_1 = F.relu(self.bn2d_1(self.conv2d_1(lf1)), inplace=True)  # (B, 64, N, n1)
        lf1_2 = F.relu(self.bn2d_2(self.conv2d_2(lf1_1)), inplace=True)  # (B, 64, N, n1)
        lf1_3 = F.relu(self.bn2d_3(self.conv2d_3(lf1_2)), inplace=True)  # (B, 128, N, n1)
        lf1_4 = F.relu(self.bn2d_4(self.conv2d_4(lf1_3)), inplace=True)  # (B, 256, N, n1)
        #lf1_5 = F.relu(self.bn2d_5(self.conv2d_5(lf1_4)), inplace=True)  # (B, 512, N, n1)

        #第二个分支
        lf2_1 = F.relu(self.bn2d_1(self.conv2d_1(lf2)), inplace=True)  # (B, C, N, n2)
        lf2_2 = F.relu(self.bn2d_2(self.conv2d_2(lf2_1)), inplace=True)
        lf2_3 = F.relu(self.bn2d_3(self.conv2d_3(lf2_2)), inplace=True)
        lf2_4 = F.relu(self.bn2d_4(self.conv2d_4(lf2_3)), inplace=True)
        #lf2_5 = F.relu(self.bn2d_5(self.conv2d_5(lf2_4)), inplace=True)  # if nbrs_num2==1

        lf1_f = torch.cat((lf1_1, lf1_2, lf1_3, lf1_4), dim=1)   # (B, C, N, n1) -> (B, 512, N, n1)
        lf1_f_d = lf1_f * rand_mask_f1
        lf1_m = lf1_f_d.max(dim=-1,keepdim=False)[0]                    # (B, C, N) -> (B, 512, N)
        #lf1_m = (lf1_m.max(dim=-1, keepdim=True)[0]).repeat(1, 1, N)  # B, 512, N, 1) -> (B, 512, N)   此处为全局特征

        lf2_f = torch.cat((lf2_1, lf2_2, lf2_3, lf2_4), dim=1)
        lf2_f_d = lf2_f * rand_mask_f2
        lf2_m = lf2_f_d.max(dim=-1, keepdim=False)[0]
        if self.nbrs_num2==1:
            lf2_m = (lf2_m.max(dim=-1, keepdim=True)[0]).repeat(1, 1, N)    # if nbrs_num2==1  此处为全局特征

        return [lf1_m, lf2_m]

if __name__ == '__main__':
    source, template = torch.rand(10, 1024, 3), torch.rand(10, 1024, 3)
    encoder = Encoder(16,1)
    ffnet = FFNet()

    src_feat = encoder(source) #(B,512,N) (B,512,N)
    tgt_feat = encoder(template)  # (B,512,N) (B,512,N)

    # sf1_m = torch.cat((sf1_m,tf1_m),dim=1)   #(B, 1024, N)
    # sf2_m = torch.cat((sf2_m,tf2_m),dim=1)
    #
    # tf1_m = torch.cat((tf1_m, sf1_m), dim=1)
    # tf2_m = torch.cat((tf2_m, sf2_m), dim=1)

    # result = ffnet(sf1_m, sf2_m)   #(B,1024,N)

    import ipdb;ipdb.set_trace()