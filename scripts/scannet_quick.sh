#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py
--dataset_name scannet \
--nqueries 256 \
--max_epoch 90 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir outputs/scannet_quick
