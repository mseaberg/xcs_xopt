#!/bin/env bash

while getopts u:e: flag
do 
    case "${flag}" in
        e) experiment=${OPTARG};;
    esac
done

source /cds/home/s/seaberg/.bashrc
source /cds/home/s/seaberg/psana_setup.sh
cd /cds/home/s/seaberg/Python/wfs_interface

if [ -z ${experiment+x} ]; then
    python run_interface.py
else
    python run_interface.py -e $experiment
fi


#ssh $username@psdev "source /cds/home/s/seaberg/psana_setup.sh; cd /cds/home/s/seaberg/Python/wfs_interface; python run_interface.py -e $experiment"
#cd /cds/home/s/seaberg/Python/wfs_interface

#python WFS_interface.py
