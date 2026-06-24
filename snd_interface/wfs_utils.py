import configparser
import os
import numpy as np

def get_immediate_subdirectories(a_dir,hutch):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and
            name[:3].lower()==hutch.lower()]

def parse_wfs_config_gui(configFile):
    print(configFile)
    config = configparser.ConfigParser()
    config.read(configFile)
    pars = {}
    pars['exp_name'] = config.get('Main','exp_name')
    pars['hutch'] = config.get('Main','hutch')
    pars['live'] = config.getboolean('Main','live')
    pars['update_events'] = config.getint('Update','update_events')
    xmin = config.getint('Processing','xmin')
    xmax = config.getint('Processing','xmax')
    ymin = config.getint('Processing','ymin')
    ymax = config.getint('Processing','ymax')
    pars['pixel'] = config.getfloat('Setup','pixel')
    pars['roi'] = [xmin,xmax,ymin,ymax]
    pars['detName'] = config.get('Setup','detName')
    pars['energy'] = config.getfloat('Main','energy')
    pars['thresh'] = config.getint('Processing','thresh')
    pars['ipm_threshold'] = config.getint('Processing','ipm_threshold')


    return pars



def parse_wfs_config(configFile):
    config = configparser.ConfigParser()
    config.read('config/'+configFile+'.cfg')
    pars = {}
    pars['exp_name'] = config.get('Main','exp_name')
    pars['hutch'] = config.get('Main','hutch')
    pars['live'] = config.getboolean('Main','live')
    pars['update_events'] = config.getint('Update','update_events')
    xmin = config.getint('Processing','xmin')
    xmax = config.getint('Processing','xmax')
    ymin = config.getint('Processing','ymin')
    ymax = config.getint('Processing','ymax')
    pars['grating_z'] = config.get('Setup','grating_z')
    pars['det_z'] = config.get('Setup','det_z')
    pars['pitch'] = config.getfloat('Setup','pitch')
    pars['z0'] = config.getfloat('Setup','z0')
    pars['zf'] = config.getfloat('Setup','zf')
    pars['pixel'] = config.getfloat('Setup','pixel')
    pars['roi'] = [xmin,xmax,ymin,ymax]
    pars['pad'] = config.getint('Processing','pad')
    pars['detName'] = config.get('Setup','detName')
    pars['energy'] = config.getfloat('Main','energy')
    pars['angle'] = config.getfloat('Setup','angle')
    pars['lineout_width'] = config.getfloat('Processing','lineout_width')
    pars['thresh'] = config.getint('Processing','thresh')
    pars['ipm_threshold'] = config.getint('Processing','ipm_threshold')
    pars['downsample'] = config.getint('Processing','downsample')

    return pars

def image_padding(N, Ndown):
    
    # expand field of view by factor sqrt(2) so that the whole image fits in the unit circle
    Nfit = np.ceil(N/Ndown*np.sqrt(2))
    # make sure Nfit is even
    if np.mod(Nfit,2) == 1:
        Nfit += 1
        
    Nfit = int(Nfit)
    # figure out how much the images need to be padded
    Npad = int((Nfit - int(N/Ndown))/2)
    
    return Nfit, Npad

def image_padding_legendre(Ndown, roi):
   
    xmin = roi[0]
    xmax = roi[1]
    ymin = roi[2]
    ymax = roi[3]

    Nx = xmax - xmin
    Ny = ymax - ymin

    Nfit_x = np.ceil(Nx/Ndown)
    Nfit_y = np.ceil(Ny/Ndown)
    # make sure Nfit is even
    if np.mod(Nfit_x,2) == 1:
        Nfit_x += 1
    if np.mod(Nfit_y,2) == 1:
        Nfit_y += 1
        
    Nfit_x = int(Nfit_x)
    Nfit_y = int(Nfit_y)
    Npad_x = int((Nfit_x - int(Nx/Ndown))/2)
    Npad_y = int((Nfit_y - int(Ny/Ndown))/2)
 
    return Nfit_x, Nfit_y, Npad_x, Npad_y
