#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

python main.py \
--dataset_name sunrgbd \
--max_epoch 1080 \
--enc_type masked \
--nqueries 128 \
--base_lr 7e-4 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--loss_giou_weight 0 \
--loss_sem_cls_weight 0.8 \
--loss_no_object_weight 0.1 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir outputs/sunrgbd_masked_ep1080
