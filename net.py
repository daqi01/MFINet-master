import torch
import torch.nn as nn
from pooling import Pooling
from ops.transform_functions import PCRNetTransform as transform
from module import FFNet, Encoder

class MFINet(nn.Module):
    def __init__(self, feature_model=Encoder(), droput=0.3, pooling='max'):
        super().__init__()
        self.feature_model = feature_model
        self.feature_fusion = FFNet()
        self.pooling = Pooling(pooling)

        self.linear = [nn.Linear(1024 * 2, 1024), nn.ReLU(),
                       nn.Linear(1024, 512), nn.ReLU(),
                       nn.Linear(512, 512), nn.ReLU(),
                       nn.Linear(512, 256), nn.ReLU()]

        # self.linear_2 = [nn.Linear(1024 * 2, 1024), nn.ReLU(),
        #                nn.Linear(1024, 1024), nn.ReLU(),
        #                nn.Linear(1024, 512), nn.ReLU(),
        #                nn.Linear(512, 512), nn.ReLU(),
        #                nn.Linear(512, 256), nn.ReLU()]

        self.linear_4d = nn.Linear(256, 4)
        self.linear_3d = nn.Linear(256, 3)

        #self.linear_7d = nn.Linear(256, 7)

        if droput > 0.0:
            self.linear.append(nn.Dropout(droput))
            #self.linear_2.append(nn.Dropout(droput))
        #self.linear.append(nn.Linear(256, 4))
        #self.linear_2.append(nn.Linear(256, 3))

        self.linear = nn.Sequential(*self.linear)
        #self.linear_2 = nn.Sequential(*self.linear_2)

    # Single Pass Alignment Module (SPAM)
    # def spam(self, template_features, source, est_R, est_t):
    #     batch_size = source.size(0)
    #
    #     self.source_features = self.pooling(self.feature_model(source))
    #
    #     y = torch.cat([template_features, self.source_features], dim=1)
    #     pose = self.linear(y)
    #     pose_4d = self.linear_4d(pose)
    #     pose_3d = self.linear_3d(pose)
    #
    #     pose_7d = torch.cat([pose_4d,pose_3d],dim=-1)
    #     pose_7d = transform.create_pose_7d(pose_7d)
    #
    #     # Find current rotation and translation.
    #     identity = torch.eye(3).to(source).view(1, 3, 3).expand(batch_size, 3, 3).contiguous()
    #     est_R_temp = transform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
    #     est_t_temp = transform.get_translation(pose_7d).view(-1, 1, 3)
    #
    #     # update translation matrix.
    #     est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
    #     # update rotation matrix.
    #     est_R = torch.bmm(est_R_temp, est_R)
    #
    #     source = transform.quaternion_transform(source, pose_7d)  # Ps' = est_R*Ps + est_t
    #     return est_R, est_t, source

    def spam(self, tf1_m, tf2_m, source, est_R, est_t):
        batch_size = source.size(0)

        #[sf1_m, sf2_m] = self.feature_model(source)
        self.src_feat = self.feature_model(source)

        tf1_m_tmp = torch.cat((tf1_m, self.src_feat[0]), dim=1)
        tf2_m_tmp = torch.cat((tf2_m, self.src_feat[1]), dim=1)

        sf1_m = torch.cat((self.src_feat[0], tf1_m), dim=1)  # (B, 1024, N)
        sf2_m = torch.cat((self.src_feat[1], tf2_m), dim=1)

        template_features = self.pooling(self.feature_fusion(tf1_m_tmp, tf2_m_tmp))  #(B,1024,N) ->(B,1024)
        source_features = self.pooling(self.feature_fusion(sf1_m, sf2_m)) #(B,1024,N) ->(B,1024)

        y = torch.cat([template_features, source_features], dim=1)
        pose = self.linear(y)
        #pose_7d = self.linear_7d(pose)

        pose_4d = self.linear_4d(pose)
        pose_3d = self.linear_3d(pose)
        pose_7d = torch.cat([pose_4d, pose_3d], dim=-1)

        pose_7d = transform.create_pose_7d(pose_7d)

        # Find current rotation and translation.
        identity = torch.eye(3).to(source).view(1, 3, 3).expand(batch_size, 3, 3).contiguous()
        est_R_temp = transform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
        est_t_temp = transform.get_translation(pose_7d).view(-1, 1, 3)

        # update translation matrix.
        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        # update rotation matrix.
        est_R = torch.bmm(est_R_temp, est_R)

        source = transform.quaternion_transform(source, pose_7d)  # Ps' = est_R*Ps + est_t
        return est_R, est_t, source, pose_7d

    # def forward(self, template, source, max_iteration=5):
    #     est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
    #     est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
    #     template_features = self.pooling(self.feature_model(template))
    #
    #     if max_iteration == 1:
    #         est_R, est_t, source = self.spam(template_features, source, est_R, est_t)
    #     else:
    #         for i in range(max_iteration):
    #             est_R, est_t, source = self.spam(template_features, source, est_R, est_t)
    #
    #     result = {'est_R': est_R,  # source -> template
    #               'est_t': est_t,  # source -> template
    #               'est_T': transform.convert2transformation(est_R, est_t),  # source -> template
    #               'r': template_features - self.source_features,
    #               'transformed_source': source}
    #     return result

    def forward(self, template, source, max_iteration=5):
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)

        #[tf1_m, tf2_m] -> tgt_feat
        tgt_feat = self.feature_model(template)
        if max_iteration == 1:
            est_R, est_t, source, pose_7d = self.spam(tgt_feat[0], tgt_feat[1], source, est_R, est_t)
        else:
            for i in range(max_iteration):
                est_R, est_t, source, pose_7d = self.spam(tgt_feat[0], tgt_feat[1], source, est_R, est_t)

        result = {'est_R': est_R,  # source -> template
                  'est_t': est_t,  # source -> template
                  'est_T': transform.convert2transformation(est_R, est_t),  # source -> template
                  #'r': template_features - self.source_features,
                  'transformed_source': source,
                  'pose_7d': pose_7d,
                  'tgt_feat':tgt_feat,
                  'src_feat':self.src_feat
                  }
        return result

if __name__ == '__main__':
    template, source = torch.rand(2, 1024, 3), torch.rand(2, 1024, 3)
    pn = Encoder()

    net = MFINet(pn)
    result = net(template, source)
    print(result['pose_7d'])
    import ipdb;ipdb.set_trace()