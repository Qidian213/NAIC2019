# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func,eval_submit
from .re_ranking import re_ranking
from .distance import low_memory_local_dist

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.scores = []
        self.feats = []
        self.local_feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def update(self, output):
        score, feat, local_feat, pid, camid, img_paths = output
        self.scores.append(score)
        self.feats.append(feat)
        self.local_feats.append(local_feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(np.asarray(img_paths))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        local_feats = torch.cat(self.local_feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]
        qlf = local_feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = np.asarray(self.img_paths[:self.num_query])

        # gallery
        gf = feats[self.num_query:]
        glf = local_feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_img_paths = np.asarray(self.img_paths[self.num_query:])

        m, n = qf.shape[0], gf.shape[0]
        
        ### global distmat
        global_distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        global_distmat.addmm_(1, -2, qf, gf.t())
        global_distmat = global_distmat.cpu().numpy()
        
        ### local distmat
        qlf = qlf.permute(0,2,1)
        glf = glf.permute(0,2,1)
        
        local_distmat = low_memory_local_dist(qlf.cpu().numpy(),glf.cpu().numpy(), aligned = True)
        dist_mat = global_distmat + 0.4*local_distmat 
        
        cmc, mAP = eval_func(dist_mat, q_pids, g_pids, q_camids, g_camids,q_img_paths, g_img_paths)

        return cmc, mAP

class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.scores = []
        self.feats = []
        self.local_feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def update(self, output):
        score, feat, local_feat, pid, camid, img_paths = output
        self.scores.append(score)
        self.feats.append(feat)
        self.local_feats.append(local_feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(np.asarray(img_paths))
        
    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        local_feats = torch.cat(self.local_feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        qlf = local_feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = np.asarray(self.img_paths[:self.num_query])

        # gallery
        gf = feats[self.num_query:]
        glf = local_feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_img_paths = np.asarray(self.img_paths[self.num_query:])

        ### local distmat
        qlf = qlf.permute(0,2,1)
        glf = glf.permute(0,2,1)
        local_distmat = low_memory_local_dist(qlf.cpu().numpy(),glf.cpu().numpy(), aligned = True)
        
        local_qq_distmat = low_memory_local_dist(qlf.cpu().numpy(),qlf.cpu().numpy(), aligned = True)
        local_gg_distmat = low_memory_local_dist(glf.cpu().numpy(),glf.cpu().numpy(), aligned = True)
        
        local_dist = np.concatenate(
            [np.concatenate([local_qq_distmat, local_distmat], axis=1),
             np.concatenate([local_distmat.T, local_gg_distmat], axis=1)],
             axis=0)

        print("Enter reranking")
        ### only global_features
 #       distmat = re_ranking(qf, gf, k1=3, k2=1, lambda_value= 0.3, wl=0.4)
        
        ### only local features
#        distmat = re_ranking(qf,gf,k1=3,k2=1,lambda_value=0.3,local_distmat=local_dist,only_local=True)
        
        ### global and local features
        distmat = re_ranking(qf,gf,k1=7,k2=2,lambda_value=0.4, wl=0.3, local_distmat=local_dist,only_local=False)
        
      #  cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids,q_img_paths, g_img_paths)
        cmc, mAP = eval_submit(distmat, q_pids, g_pids, q_camids, g_camids,q_img_paths, g_img_paths)
        return cmc, mAP
