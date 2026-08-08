#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 09/30/2024
#
# Distributed under terms of the MIT license.

""" """

import argparse
from os import path as osp

import numpy as np

from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import Meshes
from aitviewer.utils_fetal_smil import str2color
from aitviewer.viewer import Viewer


def main(exp_dir, name_list, step_idx):
    delta_x = np.array([0.5, 0, 0])
    delta_y = np.array([0, 0, 0.5])

    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    for i, name in enumerate(name_list):
        # read data
        subj_dir = osp.join(exp_dir, "subj_spec", name)
        if step_idx == 0:
            unposed_dir = osp.join(subj_dir, "init_unposed")
        else:
            unposed_dir = osp.join(subj_dir, f"{step_idx}_unposed")

        shape_his = np.load(osp.join(unposed_dir, "shape_his.npy"))
        faces = np.load(osp.join(unposed_dir, "faces.npy"))
        unposed_segm_vertex_seq = np.load(
            osp.join(unposed_dir, "segm_vertex_seq.npy"), allow_pickle=True
        )
        unposed_segm_vertex_seq = np.concatenate(unposed_segm_vertex_seq, axis=0)

        shape_his += i * delta_y
        mesh_seq = Meshes(
            shape_his, faces, color=(0.5, 0.5, 0.5, 0.5), name="BodySurfaceSMIL"
        )
        v.scene.add(mesh_seq)

        # unposed vertices
        unposed_segm_vertex_seq += i * delta_y
        ptc = PointClouds(
            points=unposed_segm_vertex_seq[None], color=(*str2color["yellow"], 0.25)
        )
        v.scene.add(ptc)

    v.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--subj_name", type=str, required=True)
    parser.add_argument("--step_idx", type=int, required=True)
    args = parser.parse_args()

    subj_name_list = args.subj_name.split(",")

    main(args.exp_dir, subj_name_list, args.step_idx)
