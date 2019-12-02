# encoding: utf-8

import torch.nn.functional as F

from .reanked_loss import RankedLoss
from .reanked_clu_loss import CRankedLoss
from .triplet_loss import TripletLoss, TripletLossAlignedReID
from .cls_loss import CrossEntropyLabelSmooth, FocalLoss_BCE, FocalLoss, sigmoid_focal_loss

def make_loss(cfg, num_classes):
    if cfg.MODEL.FPN == 'no':
        if cfg.MODEL.METRIC_LOSS_TYPE == 'ranked_loss':
            ranked_loss = RankedLoss(cfg.SOLVER.MARGIN_RANK,cfg.SOLVER.ALPHA,cfg.SOLVER.TVAL) # ranked_loss
            
        elif cfg.MODEL.METRIC_LOSS_TYPE == 'cranked_loss':
            cranked_loss = CRankedLoss(cfg.SOLVER.MARGIN_RANK,cfg.SOLVER.ALPHA,cfg.SOLVER.TVAL, cfg.SOLVER.CLU_NUM) # cranked_loss

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_loss':
            triplet = TripletLoss(cfg.SOLVER.TR_MARGIN)  # triplet loss
        else:
            print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        
        focal_loss = FocalLoss()

        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
            print("label smooth on, numclasses:", num_classes)

        if cfg.DATALOADER.SAMPLER == 'softmax':
            def loss_func(score, feat, target, local_feat):
                return F.cross_entropy(score, target)
        elif cfg.DATALOADER.SAMPLER == 'focal_loss':
            def loss_func(score, feat, target, local_feat):
                return sigmoid_focal_loss(score, target)
        elif cfg.DATALOADER.SAMPLER == 'ranked_loss':
            def loss_func(score, feat, target, local_feat):
                return ranked_loss(feat, target)[0] 
        elif cfg.DATALOADER.SAMPLER == 'cranked_loss':
            def loss_func(score, feat, target, local_feat):
                return cranked_loss(feat, target)[0] 
        elif cfg.DATALOADER.SAMPLER == 'softmax_rank':
            print('---------------softmax_rank----------------')
            def loss_func(score, feat, target, local_feat):
                if cfg.MODEL.METRIC_LOSS_TYPE == 'ranked_loss':
                    if cfg.MODEL.IF_LABELSMOOTH == 'on':
                        return xent(score, target) + cfg.SOLVER.WEIGHT*ranked_loss(feat, target) # new add by zzg, open label smooth
                    else:
                        return F.cross_entropy(score, target) + ranked_loss(feat, target)    # new add by zzg, no label smooth
                elif cfg.MODEL.METRIC_LOSS_TYPE == 'cranked_loss':
                    if cfg.MODEL.IF_LABELSMOOTH == 'on':
                        return xent(score, target) +cfg.SOLVER.WEIGHT*cranked_loss(feat, target) # new add by zzg, open label smooth
                    else:
                        return F.cross_entropy(score, target) + cranked_loss(feat, target)    # new add by zzg, no label smooth
                else:
                    print('expected METRIC_LOSS_TYPE should be triplet'
                          'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
            print('---------------softmax_triplet----------------')
            def loss_func(score, feat, target, local_feat):
                if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_loss':
                    if cfg.MODEL.IF_LABELSMOOTH == 'on':
                        return xent(score, target) + cfg.SOLVER.TR_WEIGHT*triplet(feat, target)[0]
                    else:
                        return F.cross_entropy(score, target) + cfg.SOLVER.TR_WEIGHT*triplet(feat, target)[0]
                else:
                    print('expected METRIC_LOSS_TYPE should be triplet'
                          'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        elif cfg.DATALOADER.SAMPLER == 'focal_triplet':
            print('---------------focal_triplet----------------')
            def loss_func(score, feat, target, local_feat):
                return focal_loss(score, target) + cfg.SOLVER.TR_WEIGHT*triplet(feat, target)[0]
        else:
            print('expected sampler should be softmax, ranked_loss or cranked_loss, '
                  'but got {}'.format(cfg.DATALOADER.SAMPLER))
        return loss_func
        
    else:
        print('-------------train with fpn-----------')
        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_loss':
            triplet = TripletLossAlignedReID(cfg.SOLVER.TR_MARGIN)  # triplet loss
        else:
            print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
        
        if cfg.MODEL.IF_LABELSMOOTH == 'on':
            xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
            print("label smooth on, numclasses:", num_classes)

        def loss_func(score, feat, target, local_feat):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_loss':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + cfg.SOLVER.TR_WEIGHT*triplet(feat, target, local_feat)
                    #return triplet(feat, target, local_feat)
                else:
                    return F.cross_entropy(score, target) + cfg.SOLVER.TR_WEIGHT*triplet(feat, target, local_feat)
        return loss_func


