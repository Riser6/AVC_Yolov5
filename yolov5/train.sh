#!/bin/bash
source activate yolov5
python -m torch.distributed.launch --nproc_per_node 2 train.py --batch-size 8 --sync-bn