#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --partition=kostas-compute
#SBATCH --qos=kd-med
#SBATCH --gpus=a40:1


PYTHON=$MAMBADIR/envs/ev/bin/python
SCRIPT=$SWDIR/projects/ev/scripts/train.py

cd $SWDIR/projects/ev/
$PYTHON -u $SCRIPT --batch-size 1024 --num-epochs 200 --training-root data/train_spline --validation-root data/valid_spline $@


exit 0