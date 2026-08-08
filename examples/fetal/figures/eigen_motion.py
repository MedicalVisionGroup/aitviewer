#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 11/13/2024
#
# Distributed under terms of the MIT license.

""" """

import argparse
import os
from os import path as osp

import numpy as np
import torch

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


def main(idx_pc):
    # mean pose
    mean_pose_file_path = (
        "/Users/yingchengliu/Dropbox (MIT)/code/MIT & work/fetal_pose/"
        "skel-mac/results/1031_all_v2_re_eval/evaluation/model_0/num_betas_10/"
        "train/mean_pose.npy"
    )
    mean_pose = np.load(mean_pose_file_path)

    pose_pcs_file_path = (
        "/Users/yingchengliu/Dropbox (MIT)/code/MIT & work/fetal_pose/"
        "skel-mac/results/1031_all_v2_re_eval/evaluation/model_0/num_betas_10/"
        "train/pose_pcs.npy"
    )
    pose_pcs = np.load(pose_pcs_file_path)

    pose_seq = pose_pcs[idx_pc] + mean_pose[None, :]  # (t, 69)
    print("pose_seq.shape: {}".format(pose_seq.shape))
    t = pose_seq.shape[0]

    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    smpl_layer = SMPLLayer(
        model_type="smpl",
        gender="infant",
    )

    smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=torch.tensor(pose_seq),
        betas=torch.zeros(t, 10),
        trans=torch.zeros(t, 3),
        poses_root=torch.zeros(t, 3),
    )
    v.scene.add(smpl_seq)

    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--idx_pc", type=int, default=0)
    args = parser.parse_args()
    main(args.idx_pc)
