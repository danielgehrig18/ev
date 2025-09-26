#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2GB
#SBATCH --partition=kostas-compute
#SBATCH --qos=kd-med

PYTHON=$MAMBADIR/envs/ev/bin/python
SCRIPT=$SWDIR/projects/ev/scripts/generate_data.py

cd $SWDIR/projects/ev/
$PYTHON -u $SCRIPT $@


exit 0