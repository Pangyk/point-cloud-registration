#!/bin/bash

for i in {0..0}; do
  n=$(printf %06d $i)
  echo $n
  python main.py --filename logs/test_$n.obj \
                 --pcd_filename logs/test_pcd_$n.pcd \
                 --length 150 \
                 --rad_y 30 \
                 --rad_z 35 \
                 --n_ridges 3 \
                 --visualize
done
