#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/23/2024
#
# Distributed under terms of the MIT license.

""" """

import argparse

import numpy as np
import torch

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


def get_A_pose():
    """
    Get the A pose
    """
    pose = np.zeros((23, 3))
    pose[15] = [0, 0, -0.6]  # left shoulder
    pose[16] = [0, 0, 0.6]  # right shoulder
    pose = pose.flatten()
    pose = torch.tensor(pose)
    return pose

def main(fetal_smpl_data_dict_path, idx_pc, flip_plus_minus):
    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()


    smpl_layer = SMPLLayer(
        model_type="smpl",
        gender="infant",
        fetal_smpl_data_dict_path=fetal_smpl_data_dict_path,
    )

    delta_x = np.array([0.7, 0, 0])
    delta_x = torch.tensor(delta_x)
    # delta_y = np.array([0, 0, 0.5])
    # delta_y = torch.tensor(delta_y)

    # linspace 0.1 to 2
    n_seq = 100
    stddev_seq = np.linspace(0.0, 2.0, n_seq)

    # the minus
    betas = torch.zeros(n_seq, 10)
    betas[:, idx_pc] = torch.tensor(-stddev_seq if flip_plus_minus else stddev_seq)
    smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=torch.zeros(n_seq, 3 * 23),
        betas=betas,
        trans=torch.zeros(n_seq, 3) + delta_x * (-1),
        poses_root=torch.zeros(n_seq, 3),
    )
    v.scene.add(smpl_seq)

    # the mean
    betas = torch.zeros(n_seq, 10)
    smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=torch.zeros(n_seq, 3 * 23),
        betas=betas,
        trans=torch.zeros(n_seq, 3),
        poses_root=torch.zeros(n_seq, 3),
    )
    v.scene.add(smpl_seq)

    # the plus
    betas = torch.zeros(n_seq, 10)
    betas[:, idx_pc] = torch.tensor(stddev_seq if flip_plus_minus else -stddev_seq)
    smpl_seq = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=torch.zeros(n_seq, 3 * 23),
        betas=betas,
        trans=torch.zeros(n_seq, 3) + delta_x * 1,
        poses_root=torch.zeros(n_seq, 3),
    )
    v.scene.add(smpl_seq)

    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fetal_smpl_data_dict_path", type=str, required=True)
    parser.add_argument("--idx_pc", type=int, required=True)
    parser.add_argument("--flip_plus_minus", action="store_true")
    args = parser.parse_args()

    main(args.fetal_smpl_data_dict_path, args.idx_pc, args.flip_plus_minus)
