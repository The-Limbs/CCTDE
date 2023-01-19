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

def calc_sampling_time(times):
#Calculates average time-between-samples from an array of sample times.
    t_sampling = np.mean(np.diff(times))
    return t_sampling

def calc_distance(R1,R2,z1,z2):
#Calculates the distance between two locations (R1,z1) and (R2,z2).
    distance = np.sqrt((R2-R1)**2 + (z2-z1)**2)
    return distance

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

def infer_1D_velocity(sig1,sig2,times,R1,R2,z1,z2,correlation_threshold):
    '''
    Infers velocity in one direction from a cross-correlation function. 
    Arguments: (ccf,lags,t_sampling,distance,correlation_threshold)
    Returns: velocity,correlation_max

    Parameters
    ----------
    sig1,sig2 : 1D numpy array
        two input array to be cross-correlated.
    times : 1D numpy array
        array containing times at which samples were taken. Assumed to be the same for sig1 and sig2. Assumed to be in [s]
    R1,R2,z1,z2 : floats
        R- and z-locations of sig1 and sig2. Distances expected to be in [m]
    correlation_threshold : float between 0 and 1
        defines the minimum correlation required for velocity inference. If below threshold then velocity defaults to np.nan.

    Returns
    -------
    velocity : float
        the inferred velocity [km/s].
    correlation_max : float
        the peak correlation value used to infer velocity.

    Notes
    -----
    :: if np.nan is returned, correlation threshold was not surpassed OR time-lag was equal to zero.
    '''
    # calculate ccf 
    ccf,lags = calc_ccf(sig1,sig2)
    # find the peak of the cross-correlation function
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
            #calculate unit conversion factors
            distance = calc_distance(R1,R2,z1,z2)
            t_sampling = calc_sampling_time(times)
            #calculate velocity
            velocity = 1./time_delay * (distance/t_sampling)/1000.
    else:
        # set veloity to nan if below correlation threshold
        velocity = np.nan
    return velocity,correlation_max

def analyse_consecutive_clips_1D(sig1,sig2,times,R1,R2,z1,z2,N,correlation_threshold,iterationlimit = 10000):
    '''
    This function takes two time series and splits them into consecutive, non-overlapping shorter time-series of length N. Each pair of shorter time-series is then cross-correlated and a velocities are inferred.
    It returns a one dimensional array of inferred velocities and corresponding times.

    Arguments: (sig1,sig2,times,R1,R2,z1,z2,N,correlation_threshold,iterationlimit = 10000)
    Returns: inferred_velocities,inference_times

    Parameters
    ----------
    sig1,sig2 : 1D numpy array
        two input array to be cross-correlated.
    N: integer
        the length of the individual time-series to be analysed [number of frames]
    times : 1D numpy array
        array containing times at which samples were taken. Assumed to be the same for sig1 and sig2. Assumed to be in [s]
    R1,R2,z1,z2 : floats
        R- and z-locations of sig1 and sig2. Distances expected to be in [m]
    correlation_threshold: float
        threshold of correlation below which the inferred velocity will be ignored. [between 0 and 1]

    Keyword arguments
    -----------------
    iterationlimit: integer
        maximum number of velocity inferences to make

    Returns
    -------
    inferred_velocities: 1D numpy array
        array containing all the inferred velocities. [velocity output in km/s]
    inference_times: 1D numpy array
        contains the times at which the velocity inferences were taken. Times taken as the middle of the time-series. Measured in [s].

    Notes
    -----
    :: 
    '''
    #initialise
    more_data=True
    i = 0
    inferred_velocities = np.zeros(iterationlimit)
    inferred_velocities[:] = np.nan
    inference_times = np.zeros(iterationlimit)
    inference_times[:] = np.nan
    #loop until there is no more data
    while more_data:
        #take slices of time-series
        sliced_sig1 = sig1[i:i+N]
        sliced_sig2 = sig2[i:i+N]
        sliced_times = times[i:i+N]
        #cross-correlate ts slices and infer velocity
        velocity, maxcorr = infer_1D_velocity(sliced_sig1,sliced_sig2,sliced_times,R1,R2,z1,z2,correlation_threshold)
        #store velocity in array
        inferred_velocities[int(i/N)] = velocity
        inference_times[int(i/N)] = np.mean(sliced_times)
        #move the current starting point
        i = i + N
        # abort loop if there is not enough data left in the time-series
        if i > len(sig1): more_data = False
        # abort if iterationlimit is exceeded
        if i/N > iterationlimit: more_data = False
    return inferred_velocities,inference_times

######################################################################################################################
######################################################################################################################
# scripts in development or deprecated
######################################################################################################################
######################################################################################################################


def depr_infer_2D_velocity(time_series,ref_location,spatial_seperation,correlation_threshold):
    '''

    Deprecated code!

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

