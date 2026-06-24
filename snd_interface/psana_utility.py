import h5py
import numpy as np

def save_all(comm,rank,dataSummary,arguments,fileName):

    # gather data
    for key in dataSummary.keys():
        dataSummary[key] = comm.gather(dataSummary[key])

    if rank == 0:

        # concatenate data across ranks
        for key in dataSummary.keys():
            dataSummary[key] = np.concatenate(dataSummary[key])

        # mask out events that don't exist
        mask = dataSummary['eventNums'] >= 0

        # remove empty events from all arrays
        for key in dataSummary.keys():
            dataSummary[key] = dataSummary[key][mask]

        # sort arrays based on event number
        i1 = np.argsort(dataSummary['eventNums'])
        for key in dataSummary.keys():
            dataSummary[key] = dataSummary[key][i1]

        # write hdf5 file
        with h5py.File(fileName,'w') as f:
            for key in dataSummary.keys():
                f.create_dataset(key, data=dataSummary[key])
            for key in arguments.keys():
                f.create_dataset(key, data=arguments[key])
