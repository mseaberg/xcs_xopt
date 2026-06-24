#!/bin/bash
home_path=/cds/home/s/seaberg
source $home_path/psana_setup3.sh
#source /reg/g/psdm/etc/psconda.sh -py2
cd $home_path/Python/snd_interface
python -u mpi_reply.py -b XCS -e xcsc00125 -r 1 -c config.cfg -l True

