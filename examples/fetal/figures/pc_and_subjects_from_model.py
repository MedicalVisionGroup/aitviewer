#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/15/2024
#
# Distributed under terms of the MIT license.

"""Visualize smil model in a grid"""

import argparse
import json
import math
import os
from os import path as osp

import numpy as np
import torch

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


def main(
    exp_dir,
    subj_name_list,
    step_idx,
    set_global_zero,
    flatten,
    amplify_1_2,
    magnify_diff,
):
    assert step_idx == 1

    # model
    fetal_smpl_data_dict_path = osp.join(exp_dir, "model", f"step_{step_idx}.npy")

    # subj name 2 betas
    subj_name2betas_path = osp.join(exp_dir, f"subj_name2pc_step_{step_idx}.json")
    with open(subj_name2betas_path, "r") as f:
        subj_name2betas = json.load(f)

    smpl_layer = SMPLLayer(
        model_type="smpl",
        gender="infant",
        is_rigged=False,
        fetal_smpl_data_dict_path=fetal_smpl_data_dict_path,
    )

    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    num_col = 10

    delta_x = np.array([0.6, 0, 0])
    delta_y = np.array([0, 0.6, 0])

    rate_amplify_1_2 = 1.4

    for i, name in enumerate(subj_name_list):
        # read data
        exp_subj_dir = osp.join(exp_dir, "subj_spec", name)
        exp_subj_posed_dir = osp.join(exp_subj_dir, f"{step_idx}_posed")
        tr_seq = np.load(osp.join(exp_subj_posed_dir, "transl_seq_his.npy"))[-1]
        go_seq = np.load(osp.join(exp_subj_posed_dir, "global_orient_seq_his.npy"))[-1]
        bp_seq = np.load(osp.join(exp_subj_posed_dir, "body_pose_seq_his.npy"))[-1]

        # beta
        betas = subj_name2betas[name]
        betas = np.array(betas)

        if amplify_1_2:
            # amplify pc1 and pc2
            betas[1] *= rate_amplify_1_2
            betas[2] *= rate_amplify_1_2

        if set_global_zero:
            go_seq *= 0
            tr_seq *= 0

        if flatten:
            go_seq *= 0
            tr_seq *= 0
            bp_seq *= 0

        # read subj spec shape for last step
        if step_idx == 0:
            last_unposed_folder_name = "init_unposed"
        else:
            last_unposed_folder_name = f"{step_idx}_unposed"
        last_subj_spec_shape_path = osp.join(
            exp_subj_dir, last_unposed_folder_name, "shape_his.npy"
        )
        # subj_spec_shape = np.load(last_subj_spec_shape_path)[-1]

        delta_transl = delta_x * (i % num_col) + delta_y * (i // num_col)
        this_smpl_seq = SMPLSequence(
            poses_body=bp_seq,
            smpl_layer=smpl_layer,
            betas=torch.tensor(betas, dtype=torch.float32),
            trans=tr_seq + delta_transl[None, :],
            poses_root=go_seq,
        )
        v.scene.add(this_smpl_seq)

    if magnify_diff:
        # we use the first subject as anchor,
        # and magnify the difference between the
        # first and the second subject.
        beta_first = subj_name2betas[subj_name_list[0]]
        beta_second = subj_name2betas[subj_name_list[1]]

        if amplify_1_2:
            beta_first[1] *= rate_amplify_1_2
            beta_first[2] *= rate_amplify_1_2

        diff_beta = np.array(beta_second) - np.array(beta_first)

        # ratio_magnify = 2.5
        ratio_magnify = 3
        beta_second_magnified = beta_first + diff_beta * ratio_magnify

        n = len(subj_name_list)
        delta_transl = delta_x * (n % num_col) + delta_y * (n // num_col)
        this_smpl_seq = SMPLSequence(
            poses_body=bp_seq * 0,
            smpl_layer=smpl_layer,
            betas=torch.tensor(beta_second_magnified, dtype=torch.float32),
            trans=tr_seq * 0 + delta_transl[None, :],
            poses_root=go_seq * 0,
        )
        v.scene.add(this_smpl_seq)

    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--subj_name_list", type=str, default="")
    parser.add_argument("--step_idx", type=int, default=1)
    parser.add_argument("--set_global_zero", action="store_true")
    parser.add_argument("--flatten", action="store_true")
    parser.add_argument("--amplify_1_2", action="store_true")
    parser.add_argument("--magnify_diff", action="store_true")
    args = parser.parse_args()

    subj_spec_result_dir = osp.join(args.exp_dir, "subj_spec")
    all_avail_subj_name_list = sorted(os.listdir(subj_spec_result_dir))

    if len(args.subj_name_list) == 0:
        subj_name_list = all_avail_subj_name_list
    else:
        args.subj_name_list = args.subj_name_list.split(",")
        args.subj_name_list = [sn.strip() for sn in args.subj_name_list]
        if not all(sn in all_avail_subj_name_list for sn in args.subj_name_list):
            missing_subj_name = [
                sn for sn in args.subj_name_list if sn not in all_avail_subj_name_list
            ]
            raise ValueError("Invalid subject name: {}".format(missing_subj_name))
        subj_name_list = args.subj_name_list

    if args.magnify_diff:
        assert len(subj_name_list) == 2, "Only support 2 subjects for magnify_diff"
        assert args.flatten, "flatten must be True for magnify_diff"

    main(
        args.exp_dir,
        subj_name_list,
        args.step_idx,
        args.set_global_zero,
        args.flatten,
        args.amplify_1_2,
        args.magnify_diff,
    )
