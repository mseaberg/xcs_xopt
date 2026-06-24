#!/bin/bash

#SBATCH --partition=milano
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=120
#SBATCH --output=logs/%j.log

mpirun python -u -m mpi4py.run process_xtc.py -r $1
