source /reg/g/psdm/etc/psconda.sh
`which mpirun` --oversubscribe -H daq-xcs-mon06,daq-xcs-mon07 -n 16 ./run_multiple_nodes.sh
