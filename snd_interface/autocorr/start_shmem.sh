source /reg/g/psdm/etc/psconda.sh
`which mpirun` --oversubscribe -H daq-xcs-mon11,daq-xcs-mon12 -n 16 ./mpi_driver.sh
