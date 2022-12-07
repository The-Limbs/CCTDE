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



def analyse_consecutive_clips(ts1,ts2,N,spatial_seperation,correlation_threshold,iterationlimit = 10000,plot_bool= False):
    '''
    This function takes two time series and splits them into consecutive, non-overlapping shorter time-series of length N. Each pair of shorter time-series is then cross-correlated and a velocities are inferred.
    It returns a one dimensional array of inferred velocities.

    Arguments: (ts1,ts2,N,spatial_seperation,correlation_threshold,iterationlimit = 10000,plot_bool= False)
    Returns: inferred_velocities

    Parameters
    ----------
    ts1,ts2 : 1D numpy array
        two input array to be cross-correlated.
    N: interger
        the length of the individual time-series to be analysed [number of frames]
    spatial_seperation: integer
        the spatial distance between the two time series [measured in number of channels]
    correlation_threshold: float
        threshold of correlation below which the inferred velocity will be ignored. [between 0 and 1]

    Keyword arguments
    -----------------
    iterationlimit: integer
        maximum number of velocity inferences to make
    plot_bool: boolean
        True if you want to plot the cross-correlation functions.

    Returns
    -------
    inferred_velocities: 1D numpy array
        array containing all the inferred velocities. [velocity output in px/frame]

    Notes
    -----
    :: velocity is currently estimated assuming that the spatial seperation between spatial channels/pixels is constant.
    :: This may need to be refined in the future.
    '''
    #initialise
    more_data=True
    i = 0
    inferred_velocities = np.zeros(iterationlimit)
    inferred_velocities[:] = np.nan
    #loop until there is no more data
    while more_data:
        #take slices of time-series
        sliced_ts1 = ts1[i:i+N]
        sliced_ts2 = ts2[i:i+N]
        #cross-correlate ts slices and infer velocity
        ccf,tau = calc_ccf(sliced_ts1,sliced_ts2,plot_bool=plot_bool)
        velocity, maxcorr = infer_1D_velocity(ccf,tau,spatial_seperation,correlation_threshold)
        #store velocity in array
        inferred_velocities[int(i/N)] = velocity
        #move the current starting point
        i = i + N
        # abort loop if there is not enough data left in the time-series
        if i > len(ts1): more_data = False
        # abort if iterationlimit is exceeded
        if i/N > iterationlimit: more_data = False
    return inferred_velocities

######################################################################################################################
######################################################################################################################
# core cctde functions
######################################################################################################################
######################################################################################################################

def calc_ccf(f,g,norm_bool = True,plot_bool=False):
    '''
    Returns cross-correlation function and corresponding time-delays of two one-dimenaional arrays.
    Arguments: (f,g,norm_bool = True,plot_bool=False)
    Returns: ccf,lags

    Parameters
    ----------
    f,g : 1D numpy array
        two input array to be cross-correlated.

    Keyword arguments
    -----------------
    norm_bool : boolean
        turn cross-correlation function normalisation on or off, defaults to True (on).
    plot_bool : boolean
        option to plot ccf, defaults to False.

    Returns
    -------
    ccf : 1D numpy array
        cross-correlation function of arrays f and g.
    lags : 1D numpy array
        lags corresponding to the ccf.

    Notes
    -----
    :: ccf is calculated with scipy.signal.correlate.
    :: Mode is set to 'same', meaning ccf output is same length as f
    :: Method is set to 'fft', so correlation is calculated via fft method
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

def infer_1D_velocity(ccf,lags,spatial_seperation,correlation_threshold):
    '''
    Infers velocity in one direction from a cross-correlation function and spatial seperation.
    Arguments: (ccf,lags,spatial_seperation,correlation_threshold)
    Returns: velocity,correlation_max

    Parameters
    ----------
    ccf : 1D numpy array
        Cross-correlation function.
    lags : 1D numpy array
        time-lags associated with ccf. Should be given in [frames]
    spatial_seperation : integer
        the spatial seperation between the two time-series of the ccf. Given in number of spatial channels [px].
    correlation_threshold : float between 0 and 1
        defines the minimum correlation required for velocity inference. If below threshold then velocity defaults to np.nan.

    Returns
    -------
    velocity : float
        the inferred velocity in [px/frame].
    correlation_max : float
        the peak correlation value used to infer velocity.

    Notes
    -----
    :: if np.nan is returned, correlation threshold was not surpassed OR time-lag was equal to zero.
    '''
    # find the peak of the cross-correlation funstion
    correlation_max = np.max(ccf)
    # correlation peak must exceed correlation threshold
    if correlation_max>correlation_threshold:
        #find time-delay at ccf peak
        index=np.where(np.max(ccf)==ccf)[0][0]
        time_delay = lags[index]
        #account for zero-time delay scenario
        if time_delay == 0.:
            #manually set velocity to nan
            velocity = np.nan
        else:
            #calculate velocity
            velocity = spatial_seperation/time_delay
    else:
        # set veloity to nan if below correlation threshold
        velocity = np.nan
    return velocity,correlation_max

def infer_2D_velocity(time_series,ref_location,spatial_seperation,correlation_threshold):
    '''
    Infers velocity in 2D plane at a specified reference point.
    Arguments: (time_series,ref_location,spatial_seperation,correlation_threshold)
    Returns: velocities,correlations

    Parameters
    ----------
    time_series: 3D numpy array [x-space, y-space, time]
        a time_series of spatially resolved images.
    ref_location: a pair of intergers [x_index,y_index]
        the reference coordinates in the image where the velocity is to be inferred.
    spatial_seperation : integer
        the spatial seperation between the two time-series of the ccf. Given in number of spatial channels.
    correlation_threshold : float between 0 and 1
        defines the minimum correlation required for velocity inference. If below threshold then velocity defaults to np.nan.

    Returns
    -------
    velocities: a pair of floats [x_velocity,y_velocity]
        inferred velocities at reference location.
    correlations: a pair of floats [x_correlation,y_correlation]
        peak amplitude of the correlation function used for velocity inference.

    Notes
    -----
    :: In the current implementation, the selection of the locations is a bit illogical.
       The reference location should be centered between the locations of the timeseries.
       Should be changed before application to experiment.
    '''
    #select the timeseries
    i,j = ref_location
    ref = time_series[i,j,:]
    x_ts = time_series[i+spatial_seperation,j,:]
    y_ts = time_series[i,j+spatial_seperation,:]
    #calculate velocities
    velocities = np.empty(2)
    correllations = np.empty(2)
    x_ccf,x_lags = calc_ccf(ref,x_ts)
    x_vel,x_correlation=infer_1D_velocity(x_ccf,x_lags,spatial_seperation,correlation_threshold)
    y_ccf,y_lags = calc_ccf(ref,y_ts)
    y_vel,y_correlation=infer_1D_velocity(y_ccf,y_lags,spatial_seperation,correlation_threshold)
    velocities = (x_vel,y_vel)
    correlations = (x_correlation,y_correlation)
    return velocities,correlations
