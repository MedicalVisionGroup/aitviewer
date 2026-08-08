#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 01/30/2025
#
# Distributed under terms of the MIT license.

""" """

import argparse
import os
from os import path as osp

import numpy as np
import torch
from tqdm import tqdm

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


def _get_data(exp_dir, subj_name):
    subj_dir = osp.join(exp_dir, "subj_spec", subj_name, "init_posed")

    transl_seq = np.load(osp.join(subj_dir, "transl_seq_his.npy"))[-1]
    global_orient_seq = np.load(osp.join(subj_dir, "global_orient_seq_his.npy"))[-1]
    body_pose_seq = np.load(osp.join(subj_dir, "body_pose_seq_his.npy"))[-1]
    beta = np.load(osp.join(subj_dir, "beta_his.npy"))[-1]

    return transl_seq, global_orient_seq, body_pose_seq, beta


def main(num_betas=10, fetal_smpl_data_dict_path=None):
    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    smpl_layer = SMPLLayer(
        model_type="smpl",
        gender="infant",
        num_betas=num_betas,
        fetal_smpl_data_dict_path=fetal_smpl_data_dict_path,
    )

    # color option 1: rgb = (180, 149, 133)
    color1 = torch.tensor([180, 149, 133, 255]) / 255.0

    num_col = 32
    delta_x = np.array([0.5, 0, 0])
    delta_y = np.array([0, 0, 0.5])

    cnt = 0
    i_frame = 0
    subj_name = "MAP-B307"

    ############
    #  before  #
    ############

    exp_dir = (
        "./results/0125_rebuttal_all/evaluation/model_1/num_betas_04/visualization"
    )
    transl_seq, global_orient_seq, body_pose_seq, beta = _get_data(exp_dir, subj_name)
    tr = transl_seq[i_frame : i_frame + 1]
    gl = global_orient_seq[i_frame : i_frame + 1]
    bp = body_pose_seq[i_frame : i_frame + 1]

    delta_transl = delta_x * (cnt % num_col) + delta_y * (cnt // num_col)
    this_smpl_seq = SMPLSequence(
        poses_body=bp,
        smpl_layer=smpl_layer,
        betas=beta,
        trans=tr + delta_transl[None, :],
        poses_root=gl,
        color=color1,
        is_rigged=False,
    )
    v.scene.add(this_smpl_seq)
    cnt += 1

    ###########
    #  after  #
    ###########

    exp_dir = (
        "./results/0125_rebuttal_all/evaluation_0127_2/model_1/num_betas_04/visualization"
    )
    transl_seq, global_orient_seq, body_pose_seq, beta = _get_data(exp_dir, subj_name)
    tr = transl_seq[i_frame : i_frame + 1]
    gl = global_orient_seq[i_frame : i_frame + 1]
    bp = body_pose_seq[i_frame : i_frame + 1]
   
    delta_transl = delta_x * (cnt % num_col) + delta_y * (cnt // num_col)
    this_smpl_seq = SMPLSequence(
        poses_body=bp,
        smpl_layer=smpl_layer,
        betas=beta,
        trans=tr + delta_transl[None, :],
        poses_root=gl,
        color=color1,
        is_rigged=False,
    )
    v.scene.add(this_smpl_seq)
    cnt += 1

    # render
    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fetal_smpl_data_dict_path", default=None, type=str)
    parser.add_argument("--num_betas", default=10, type=int)
    args = parser.parse_args()

    main(args.num_betas, args.fetal_smpl_data_dict_path)
