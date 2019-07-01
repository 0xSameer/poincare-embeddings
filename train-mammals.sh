#!/bin/sh
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python3 embed.py \
       -dim 10 \
       -lr 0.3 \
       -epochs 300 \
       -negs 50 \
       -burnin 20 \
       -ndproc 1 \
       -manifold poincare \
       -dset wordnet/mammal_closure.csv \
       -checkpoint mammals_poincare_dim10.pth \
       -batchsize 10 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -gpu -1 \
       -train_threads 1
