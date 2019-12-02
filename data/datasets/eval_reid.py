# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import os
import numpy as np
import shutil
import cv2
import json
import shutil

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids,q_img_paths, g_img_paths, max_rank=210):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_submit(distmat, q_pids, g_pids, q_camids, g_camids,q_img_paths, g_img_paths, max_rank=200):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    result=dict() 

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    
    for q_idx in range(num_q):
        q_img_path = q_img_paths[q_idx] ## ./data/sz_reid/query/5440_c0s0_606505321_00.png
        q_dst_dir = '/home/zhangzhiguang/dataset/shen_reid/test_result/' + q_img_path.split('/')[-1].split('_')[0]
        if not os.path.exists(q_dst_dir):
            os.makedirs(q_dst_dir)
        
        q_dst_path = q_dst_dir + '/' + q_img_path.split('/')[-1]
        shutil.copyfile(q_img_path, q_dst_path)
        
        q_img_name = q_img_path.split('/')[-1].split('_')[-2] + '.png'
        result[q_img_name] = []
        
        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            g_img_path = g_img_paths[g_idx]
            
            if rank_idx <=5:
                g_dst_path = q_dst_dir + '/' + g_img_path.split('/')[-1]
                shutil.copyfile(g_img_path, g_dst_path)
            
            g_img_name = g_img_path.split('/')[-1].split('_')[-2] + '.png'
            result[q_img_name].append(g_img_name)
            
            rank_idx += 1
            if rank_idx > max_rank:
                break

    with open('./result.json', 'w') as f:
        json.dump(result, f)
        f.close()

    return 1,2

