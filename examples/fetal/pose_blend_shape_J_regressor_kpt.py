#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 10/05/2024
#
# Distributed under terms of the MIT license.

""" """

import argparse
import os
from os import path as osp

import numpy as np

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import Meshes, Skeletons
from aitviewer.utils_fetal_smil import kpt_skeleton, str2color
from aitviewer.viewer import Viewer


def main(data_dir, exp_dir, step_idx):
    # read gt data
    file_name_list = sorted(os.listdir(data_dir))
    segm_vertex_seq_list = []  # list of list of (V, 3)
    segm_faces_seq_list = []  # list of list of (F, 3)
    kpt_seq_list = []  # (N_subj, T, KPT, 3)
    for name in file_name_list:
        segm_vertex_seq = np.load(
            osp.join(data_dir, name, "segm_vertex_seq.npy"), allow_pickle=True
        )
        segm_vertex_seq_list.append(segm_vertex_seq)
        segm_faces_seq = np.load(
            osp.join(data_dir, name, "segm_faces_seq.npy"), allow_pickle=True
        )
        segm_faces_seq_list.append(segm_faces_seq)
        kpt_seq = np.load(osp.join(data_dir, name, "kpt_seq.npy"))
        kpt_seq_list.append(kpt_seq)

    def _flat_list(ll):
        flat_list = []
        for l in ll:
            flat_list.extend(l)
        return flat_list

    # concatenate all subjects
    segm_faces_seq_list = _flat_list(segm_faces_seq_list)
    segm_vertex_seq_list = _flat_list(segm_vertex_seq_list)
    kpt_seq_list = _flat_list(kpt_seq_list)
    total_T = len(kpt_seq_list)
    print("total_T: {}".format(total_T))

    # read pred data
    population_dir = osp.join(exp_dir, "population")
    result_dir = osp.join(
        population_dir, f"{step_idx}_pose_blend_shape_J_regressor_kpt"
    )

    # pred kpt history and pred shape history
    pred_kpt_list_his = np.load(
        osp.join(result_dir, "pred_kpt_list_his.npy")
    )  # (N_history, N_log_frame, N_KPT, 3)
    pred_shape_list_his = np.load(
        osp.join(result_dir, "pred_shape_list_his.npy")
    )  # (N_history, N_log_frame, V, 3)
    n_history = pred_kpt_list_his.shape[0]

    print(f"pred_kpt_list_his.shape: {pred_kpt_list_his.shape}")
    print(f"pred_shape_list_his.shape: {pred_shape_list_his.shape}")

    # check history: does it change over time at all? 
    # pred_kpt_list_his_sum = pred_kpt_list_his.reshape(n_history, -1).sum(axis=1)
    # print("pred_kpt_list_his_sum: {}".format(pred_kpt_list_his_sum))
    # this history file is not working.

    # n_log_frames are picked with even space among total T
    n_log_frame = pred_kpt_list_his.shape[1]
    print("n_log_frame: {}".format(n_log_frame))
    idx_log_frame = np.linspace(
        0, total_T - 1, n_log_frame, dtype=int
    )  # (N_log_frame,)

    smpl_layer = SMPLLayer(model_type="smpl", gender="infant")
    smpl_faces = smpl_layer.faces.numpy()  # (N_faces, 3)

    # create viewer
    Viewer.window_type = "pyqt6"
    v = Viewer()

    delta_x = np.array([0.5, 0, 0])
    delta_y = np.array([0, 0, 0.5])
    delta_z = np.array([0, 0.5, 0])

    for i, idx_frame in enumerate(idx_log_frame):

        delta = delta_x * i + delta_y * 0 + delta_z * 0
        delta = delta[None, None, :]

        # gt data
        kpt = kpt_seq_list[idx_frame][None]  # (1, KPT, 3)
        _color = (*str2color["tab_blue"], 1.0)
        ptc = Skeletons(
            kpt + delta,
            kpt_skeleton,
            gui_affine=False,
            color=_color,
            name=f"kpt_{idx_frame:02d}",
        )
        v.scene.add(ptc)

        shape = segm_vertex_seq_list[idx_frame][None]  # (1, V, 3)
        shape_faces = segm_faces_seq_list[idx_frame]
        _color = (*str2color["gray"], 0.8)
        mesh = Meshes(
            shape + delta,
            shape_faces,
            color=_color,
            name=f"shape_{idx_frame:02d}",
        )
        v.scene.add(mesh)

        # pred data
        delta_pred = delta

        pred_kpt = pred_kpt_list_his[:, i]  # (N_his, KPT, 3)
        print("pred_kpt.shape: {}".format(pred_kpt.shape))
        _color = (*str2color["tab_red"], 1.0)
        skel = Skeletons(
            pred_kpt + delta_pred,
            kpt_skeleton,
            gui_affine=False,
            color=_color,
            name=f"Keypoints_{idx_frame:02d}",
        )
        v.scene.add(skel)

        pred_shape = pred_shape_list_his[:, i]  # (N_his, V, 3)
        print("pred_shape.shape: {}".format(pred_shape.shape))
        print("delta_pred.shape: {}".format(delta_pred.shape))
        _color = (*str2color["tab_red"], 0.8)
        mesh = Meshes(
            pred_shape + delta_pred,
            smpl_faces,
            color=_color,
            name=f"Shape_{idx_frame:02d}",
        )
        v.scene.add(mesh)


    v.run() 
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--step_idx", type=int, default=1)
    args = parser.parse_args()

    main(args.data_dir, args.exp_dir, args.step_idx)
