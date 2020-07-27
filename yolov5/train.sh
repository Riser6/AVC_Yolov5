#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 2 train.py --batch-size 8 --notest
