import numpy as np
import matplotlib.pyplot as plt

def calc_pitch(lineout,fc,factor):
    # get lineout length
    N = np.size(lineout)

    # calculate spatial frequencies
    dfx = 1./N

    fx = np.linspace(0,N-1,N,dtype=float)*dfx

    mask0 = (fx-fc)**2<(fc/2)**2

    F1 = np.fft.fft(lineout)

    x_fft = F1*mask0

    #print(np.sum(np.abs(mask0)))

    x_peak = np.argmax(np.abs(x_fft)*mask0)

    x_int = x_peak*dfx

    #print(x_peak)

    Nx1 = np.size(x_fft)
    
    # shift peak to zero
    #x_shift = np.fft.fftshift(np.pad(x_fft[int(x_peak-int(x_peak/2/factor)):int(x_peak+int(x_peak/2/factor))],int(Nx1/2),'constant'))
    x_shift = x_fft[int(x_peak-int(x_peak/2/factor)):int(x_peak+int(x_peak/2/factor))]
    
    Nx = np.size(x_shift)
    
    dx_prime = 1./Nx/dfx

    x_prime = np.linspace(-Nx/2,Nx/2-1,Nx,dtype=float)*dx_prime

    #x_filt = np.fft.ifft(x_shift)[int(Nx/2-3*Nx/8)+1:int(Nx/2+3*Nx/8)-1]

    x_filt = np.conj(np.fft.ifft(x_shift))

    x_filt = x_filt[2:-2]
    x_prime = x_prime[2:-2]
    px = np.polyfit(x_prime,np.unwrap(np.angle(x_filt)),1)

    #x_prime = x_prime[int(Nx/2-3*Nx/8)+1:int(Nx/2+3*Nx/8)-1]

    #px = np.polyfit(x_prime,np.unwrap(np.angle(x_filt)),1)

    residual = np.unwrap(np.angle(x_filt)) - px[1] - px[0]*x_prime

    x_centroid = x_int+px[0]/2/np.pi

    x_pitch = 1./x_centroid

    return x_pitch, residual, x_prime


def calc_pitch_vis(lineout,fc,factor):
    # get lineout length
    N = np.size(lineout)

    # calculate spatial frequencies
    dfx = 1./N

    fx = np.linspace(0,N-1,N,dtype=float)*dfx

    mask0 = (fx-fc)**2<(fc/2)**2

    F1 = np.fft.fft(lineout)

    zeromask = fx<fc/2
    zeromask = np.logical_or(zeromask, fx>1-fc/2)


    zero_order = F1*zeromask

    x_fft = F1*mask0

    #plt.figure()
    #plt.plot(np.abs(x_fft))
    #plt.plot(np.abs(zero_order))
    #plt.show()

    x_vis = np.max(np.abs(x_fft))/np.max(np.abs(zero_order))*2

    vis2 = (np.max(lineout)-np.min(lineout))/(np.max(lineout)+np.min(lineout))

    #print(np.sum(np.abs(mask0)))

    x_peak = np.argmax(np.abs(x_fft)*mask0)

    x_int = x_peak*dfx

    #print(x_peak)

    Nx1 = np.size(x_fft)
    
    # shift peak to zero
    #x_shift = np.fft.fftshift(np.pad(x_fft[int(x_peak-int(x_peak/2/factor)):int(x_peak+int(x_peak/2/factor))],int(Nx1/2),'constant'))
    x_shift = np.fft.fftshift(x_fft[int(x_peak-int(x_peak/2/factor)):int(x_peak+int(x_peak/2/factor))])
    
    Nx = np.size(x_shift)
    
    dx_prime = 1./Nx/dfx

    x_prime = np.linspace(-Nx/2,Nx/2-1,Nx,dtype=float)*dx_prime

    #x_filt = np.fft.ifft(x_shift)[int(Nx/2-3*Nx/8)+1:int(Nx/2+3*Nx/8)-1]

    x_filt = np.conj(np.fft.ifft(x_shift))

    x_filt = x_filt[2:-2]
    x_prime = x_prime[2:-2]
    px = np.polyfit(x_prime,np.unwrap(np.angle(x_filt)),1)

    #x_prime = x_prime[int(Nx/2-3*Nx/8)+1:int(Nx/2+3*Nx/8)-1]

    #px = np.polyfit(x_prime,np.unwrap(np.angle(x_filt)),1)

    residual = np.unwrap(np.angle(x_filt)) - px[1] - px[0]*x_prime

    x_centroid = x_int-px[0]/2/np.pi

    x_pitch = 1./x_centroid

    return x_pitch, residual, x_prime, x_vis, vis2

