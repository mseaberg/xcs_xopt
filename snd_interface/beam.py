import numpy as np
import numpy.fft as fft
#import pyfftw.interfaces.numpy_fft as fft


class Beam:

    def __init__(self, input, dx, lambda0):

        self.input = input
        self.wave = np.copy(input)
        self.dx = dx
        self.lambda0 = lambda0
        self.N, self.M = input.shape
        x = np.linspace(-self.M / 2.0 * dx, (self.M / 2.0 - 1) * dx, self.M)
        y = np.linspace(-self.N / 2.0 * dx, (self.N / 2.0 - 1) * dx, self.N)
        self.x, self.y = np.meshgrid(x, y)
        self.k0 = 2.0 * np.pi / lambda0

        fxMax = 1.0 / (2.0 * dx)
        dfx = fxMax / self.M
        fx = np.linspace(-fxMax, fxMax - dfx, self.M)
        dfy = fxMax / self.N
        fy = np.linspace(-fxMax, fxMax - dfy, self.N)
        self.fx, self.fy = np.meshgrid(fx, fy)

        self.phiProp = self.k0 * (np.sqrt(1.0 - (self.lambda0 * self.fy)**2 - (self.lambda0 * self.fx)**2 ))

        self.filter1 = self.fx ** 2 + self.fy ** 2 < (1.0 / self.lambda0) ** 2

    @staticmethod
    def NFFT(input):
        return fft.fftshift(fft.fft2(fft.ifftshift(input)))

    @staticmethod
    def INFFT(input):
        return fft.fftshift(fft.ifft2(fft.ifftshift(input)))


    def beam_prop(self,dz):

        G = Beam.NFFT(self.wave)
        G = G*np.exp(1j*self.phiProp*dz)*self.filter1
        self.wave = Beam.INFFT(G)

        return self.wave

    def reset(self):
        self.wave = self.input

        return self.wave

    def multiply_screen(self,screen):

        self.wave = self.wave*screen
        return self.wave

    def farField(self,zD):

        phase = np.exp(1j*np.pi/self.lambda0/zD*(self.x**2+self.y**2))
        self.x = self.fx*self.lambda0*zD
        self.y = self.fy*self.lambda0*zD

        dfx = 1/(2*np.max(self.x))

        self.wave = Beam.NFFT(self.wave*phase)

        return self.wave

    def backPropagate(self,zF,defocus):

        phase = np.exp(1j*np.pi/self.lambda0*(self.xD**2+self.yD**2)*(1.0/zF - 1.0/(zF-defocus)))


        back = Beam.INFFT(self.waveFarField*phase)
