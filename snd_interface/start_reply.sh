#!/bin/bash
source /reg/g/psdm/etc/psconda.sh
echo $(hostname)

# path to home directory
home_path=/reg/neh/home/seaberg
# submit batch job
ssh psana "source $home_path/psana_setup.sh; cd $home_path/Python/wfs_interface; bsub -q psnehprioq -n 16 -o log/%J.log mpirun python mpi_reply.py -b $1 -e $2 -r $3 -s $4 -c $5" &
