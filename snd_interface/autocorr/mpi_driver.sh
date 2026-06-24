#!/bin/bash
# run like this:
# `which mpirun` --oversubscribe -H daq-xcs-mon11,daq-xcs-mon12 -n 8 ./mpi_driver.sh
source /reg/g/psdm/etc/psconda.sh
cd /cds/home/s/seaberg/Python/snd_interface/autocorr
# cd /reg/neh/operator/xppopr/live_data/xppx40318
python -u mpi_driver.py shmem=psana.0:stop=no epix_8
# python -u mpi_driver.py exp=xppx40318:run=234,236:smd epix_alc
#mpirun -n 2 python -u mpi_driver.py exp=xcslx6920:run=55:smd epix_2
