import numpy as np
from my_funcs import *
import warnings
import numpy.ma as ma
import matplotlib.cm as cm
import multiprocessing as mp
from scipy import signal
from scipy.signal import butter,filtfilt
from scipy.ndimage import gaussian_filter


######################################################################################################################
######################################################################################################################
# helper functions
######################################################################################################################
######################################################################################################################

def calc_norm_factor(f,g):
#calculates normalisation factor for ccf
    #set precision to reduce overflow chance
    f = f.astype(np.float32)
    g = g.astype(np.float32)
    #calculate sumsquares
    sumsquare_f= (np.sum((f)**2))
    sumsquare_g= (np.sum((g)**2))
    norm_factor = np.sqrt(sumsquare_f*sumsquare_g)
    return norm_factor

######################################################################################################################
######################################################################################################################
# core cctde functions
######################################################################################################################
######################################################################################################################

def calc_ccf(f,g,norm_bool = True,plot_bool=False):
    '''
    -----------------------------------------------------------------------------------------------
    Returns cross-correlation function and corresponding time-delays of two one dimenaional arrays

    :param f,g:          one dimensional arrays, usually time-series
    :type f,g:           expecting 1D numpy arrays
    :param norm_bool:    should the ccf be normalised?, defaults to True
    :type norm_bool:     boolean
    :param plot_bool:    should the ccf be plotted?, defaults to False
    :type plot_bool:     boolean
    :return ccf,lags:     cross-correlation function and corresponding time-delays
    :rtype ccf,lags:      1D numpy arrays
    ..notes::            ccf is calculated with scipy.signal.correlate.
                         Mode is set to 'same', meaning ccf output is same length as f
                         Method is set to 'fft', so correlation is calculated via fft method
    -----------------------------------------------------------------------------------------------
    '''
    #zero center the signals
    f = f - np.mean(f)
    g = g - np.mean(g)
    #calculate unnormalised ccf and lags
    ccf = signal.correlate(g,f,mode='same',method='fft')
    N = len(ccf)
    lags = np.linspace(-N//2,N//2-1,N)
    if norm_bool==True:
        #normalise entire ccf based on ccf rms
        norm_factor = calc_norm_factor(f,g)
        ccf = np.divide(ccf,norm_factor)
        #normalise each ccf value base on length array overlap (which varies with the lag)
        for i in range(len(ccf)):
            lag = lags[i]
            lag_norm = N/(N-np.abs(lag))
            ccf[i] = ccf[i]*lag_norm
    #plot ccf
    if plot_bool==True:
        fig,ax=plt.subplots(3)
        ax[0].plot(lags,ccf,'.',ls='-')
        ax[0].set_xlabel('time delay')
        ax[0].set_ylabel('ccf')
        ax[0].set_title('ccf')
        ax[1].plot(f,'.',ls='-')
        ax[1].set_title('f')
        ax[2].plot(g,'.',ls='-')
        ax[2].set_title('g')
        fig.tight_layout()
        plt.show()
    return ccf,lags
