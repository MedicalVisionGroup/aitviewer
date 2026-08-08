#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/23/2024
#
# Distributed under terms of the MIT license.

"""

"""
import argparse

import numpy as np
import torch

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer


def main(fetal_smpl_data_dict_path):
    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    smpl_layer = SMPLLayer(
        model_type="smpl",
        gender="infant",
        fetal_smpl_data_dict_path=fetal_smpl_data_dict_path,
    )

    delta_x = np.array([0.5, 0, 0])
    delta_x = torch.tensor(delta_x)
    delta_y = np.array([0, 0, 0.5])
    delta_y = torch.tensor(delta_y)

    # linspace -1. to 1.
    n_seq = 100
    stddev_seq = np.linspace(-1., 1., n_seq)

    # first three principal component
    for idx_pc in range(3):
        betas = torch.zeros(n_seq, 10)
        betas[:, idx_pc] = torch.tensor(stddev_seq)
        smpl_seq = SMPLSequence(
            smpl_layer=smpl_layer,
            poses_body=torch.zeros(n_seq, 3 * 23),
            betas=betas,
            trans=torch.zeros(n_seq, 3) + delta_x * idx_pc,
            poses_root=torch.zeros(n_seq, 3),
        )
        v.scene.add(smpl_seq)

    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--fetal_smpl_data_dict_path", type=str, required=True)
    args = parser.parse_args()

    main(args.fetal_smpl_data_dict_path)
