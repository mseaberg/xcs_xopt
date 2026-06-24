# run these commands in the "amoh1315" directory

# mpirun -n 2 python mpi_driver.py  exp=xpptut15:run=54 cspad -n 10
# in batch:
# bsub -q psanaq -n 2 -o %J.log -a mympi python mpi_driver.py exp=xpptut15:run=54

from data_processing import *
import wfs_utils


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'
numClients = size-1

import argparse
#import ConfigParser
parser = argparse.ArgumentParser()
parser.add_argument("-b","--hutch", help="hutch name",type=str)
parser.add_argument("-e","--experiment", help="experiment number",type=str)
parser.add_argument("-r","--run", help="run number from DAQ")
parser.add_argument("-n","--noe",help="number of events, all events=0",default=-1, type=int)
parser.add_argument("-c","--config",help="config file name",default='wfs',type=str)
parser.add_argument("-s","--server",help="server name",type=str)
parser.add_argument("-l","--live",help="run live",type=bool,default=False)
args = parser.parse_args()

pars = wfs_utils.parse_wfs_config_gui(args.config)
# parse config file
#config = ConfigParser.ConfigParser()
#config.read('config/'+args.config+'.cfg')
#pars = {}
#pars['exp_name'] = config.get('Main','exp_name')
#pars['hutch'] = config.get('Main','hutch')
#pars['live'] = config.getboolean('Main','live')
#pars['updateEvents'] = config.getint('Update','updateEvents')
#xmin = config.getint('Processing','xmin')
#xmax = config.getint('Processing','xmax')
#ymin = config.getint('Processing','ymin')
#ymax = config.getint('Processing','ymax')
#pars['grating_z'] = config.get('Setup','grating_z')
#pars['det_z'] = config.get('Setup','det_z')
#pars['pitch'] = config.getfloat('Setup','pitch')
#pars['z0'] = config.getfloat('Setup','z0')
#pars['zf'] = config.getfloat('Setup','zf')
#pars['pixel'] = config.getfloat('Setup','pixel')
#pars['roi'] = [xmin,xmax,ymin,ymax]
#pars['pad'] = config.getint('Processing','pad')
#pars['detName'] = config.get('Setup','detName')
#pars['energy'] = config.getfloat('Main','energy')
#pars['angle'] = config.getfloat('Setup','angle')
#pars['lineout_width'] = config.getfloat('Processing','lineout_width')
#pars['fraction'] = config.getfloat('Processing','fraction')
#



if rank==0:
    runmaster(numClients,args,pars,comm,rank,size)
else:
    runclient(args,pars,comm,rank,size)

MPI.Finalize()
