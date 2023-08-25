import numpy as np
from my_funcs import *
import warnings
import numpy.ma as ma
import matplotlib.cm as cm
import matplotlib
import multiprocessing as mp
from scipy import signal
from scipy.signal import butter,filtfilt
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import RANSACRegressor,LinearRegression
from scipy.optimize import curve_fit
import itertools

######################################################################################################################
######################################################################################################################
# two-point helper functions
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

def reverse_direction_check(i1,i2,z1,z2):
    if i2>i1:
        if z2>z1:
            reflect_bool = False
        elif z1>z2:
            reflect_bool = True
        else:
            print('zero error. abort.')
            print(1/0)
    elif i1>i2:
        if z2>z1:
            reflect_bool = False
        elif z1>z2:
            reflect_bool = True
        else:
            print('zero error. abort.')
            print(1/0)
    elif i1==i2:
        reflect_bool = False
    return reflect_bool

def calc_corr_threshold(N,tolerance):
    rms_ccfs = []
    iteration_limit = 1000000
    for i in range(iteration_limit):
        sigA = np.random.normal(size=N)
        sigB = np.random.normal(size=N)
        ccf,lags = calc_ccf(sigA,sigB)
        rms_ccf = np.sqrt(np.nanmean(ccf**2))
        rms_ccfs.append(rms_ccf)
        std_err = np.nanstd(rms_ccfs)/len(rms_ccfs)
        if i>100:
            if std_err <tolerance:
                break
        if i == iteration_limit-1:
            print('Err: iteration limit reached')
    print('correlation threshold={0:.2f}'.format(np.nanmean(rms_ccfs)))
    return np.nanmean(rms_ccfs)

######################################################################################################################
######################################################################################################################
# core two-point cctde functions
######################################################################################################################
######################################################################################################################

def calc_ccf(f,g,norm_bool = True,plot_bool=False,overlap_mode = 'same',method='fft'):
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
    mode : string
        can change overlap mode of signal.correlate ['same','full','valid']

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
    f = (f - np.nanmean(f))/np.nanstd(f)
    g = (g - np.nanmean(g))/np.nanstd(g)
    #calculate unnormalised ccf and lags
    ccf = signal.correlate(g,f,mode=overlap_mode,method=method)
    N = len(ccf)
    lags = np.linspace(-N//2,N//2-1,N)
    if norm_bool==True:
        #normalise entire ccf based on ccf rms
        norm_factor = calc_norm_factor(f,g)
        ccf = np.divide(ccf,norm_factor)
        if overlap_mode=='same':
            #normalise each ccf value base on length array overlap (which varies with the lag)
            for i in range(len(ccf)):
                lag = lags[i]
                lag_norm = N/(N-np.abs(lag))
                ccf[i] = ccf[i]/lag_norm
    #plot ccf
    if plot_bool==True:
        index=np.where(np.max(ccf)==ccf)[0][0]
        time_delay = lags[index]
        print('time delay: {0} frames'.format(time_delay))
        fig,ax=plt.subplots(4,figsize=(8,8))
        ax[0].plot(lags,ccf,'.',ls='-')
        ax[0].set_xlabel('time delay')
        ax[0].set_ylabel('ccf')
        ax[0].set_title('ccf')
        ax[1].plot(f*g[N//2:N+N//2-1],'.',ls='-')
        ax[1].set_title('zero-lag ccf integrand')
        ax[2].plot(f,'.',ls='-')
        ax[2].set_title('f')
        ax[3].plot(g,'.',ls='-')
        ax[3].set_title('g')
        if overlap_mode=='valid':
            ax[3].set_xlim(N//2,N+N//2)
        fig.tight_layout()
        np.save('f',f)
        np.save('g',g)
        plt.show()
    return ccf,lags

def infer_two_point_velocity(sig1,sig2,times,R1,R2,z1,z2,correlation_threshold,mode='same',plot_bool=False,return_ccf=False):
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

    Keyword arguments:
    ------------------
    plot_bool: boolean
        should the CCF be plotted?
        WARNING! make sure you're not inside several nested loops :)
        
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
    ccf,lags = calc_ccf(sig1,sig2,plot_bool=plot_bool,overlap_mode=mode)
    # check if ccf is empty
    if len(ccf) == 0:
        velocity = np.nan
        correlation_max = 0.
    else:
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
    if plot_bool==True:
        print('Velocity: {0}km/s \n Time: {1}s \n Correlation: {2}'.format(velocity,np.mean(times),correlation_max))
    if return_ccf==False:
        return velocity,correlation_max
    elif return_ccf==True:
        return velocity,correlation_max,ccf,lags
    else:
        print('error')
        print(1/0)

def analyse_consecutive_clips_two_point(sig1,sig2,times,R1,R2,z1,z2,N,stepsize,correlation_threshold,iterationlimit = 10000000,mode='same',plot_bool=False):
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
    plot_bool: boolean
        should the CCF be plotted?
        WARNING! make sure you're not inside several nested loops :)

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
    arr_length = int(len(sig1)/stepsize)+1
    inferred_velocities = np.full(arr_length,np.nan)
    inferred_correlations = np.full(arr_length,np.nan)
    inference_times = np.full(arr_length,np.nan)
    #loop until there is no more data
    while more_data:
        # calculate ccf which allows nan/zero padding
        if mode == 'same':
            #take slices of time-series
            sliced_sig1 = sig1[i:i+N]
            sliced_sig2 = sig2[i:i+N]
            sliced_times = times[i:i+N]
            #cross-correlate ts slices and infer velocity
            velocity, maxcorr = infer_two_point_velocity(sliced_sig1,sliced_sig2,sliced_times,R1,R2,z1,z2,correlation_threshold,mode=mode,plot_bool=plot_bool)
            #store velocity in array
            inferred_velocities[int(i/stepsize)] = velocity
            inferred_correlations[int(i/stepsize)] = maxcorr
            inference_times[int(i/stepsize)] = np.mean(sliced_times)
        # calculate ccf which doesn't allow padding
        elif mode == 'valid':
            #exclude edge regions
            if i <N//2:
                velocity, maxcorr = np.nan,np.nan
                inference_time = np.nan
            elif i >arr_length-N//2-N-1:
                velocity, maxcorr = np.nan,np.nan
                inference_time = np.nan
            else:
                #take slices of time-series
                sliced_sig1 = sig1[i:i+N]
                sliced_sig2 = sig2[i-N//2:i+N+N//2]
                sliced_times = times[i:i+N]
                inference_time = np.nanmean(sliced_times)
                #cross-correlate ts slices and infer velocity
                velocity, maxcorr = infer_two_point_velocity(sliced_sig1,sliced_sig2,sliced_times,R1,R2,z1,z2,correlation_threshold,mode=mode,plot_bool=plot_bool)
            #store velocity in array
            inferred_velocities[int(i/stepsize)] = velocity
            inferred_correlations[int(i/stepsize)] = maxcorr
            inference_times[int(i/stepsize)] = inference_time
        #move the current starting point
        i = i + stepsize
        # abort loop if there is not enough data left in the time-series
        if i+N >= len(sig1): more_data = False
        # abort if iterationlimit is exceeded
        if i/N > iterationlimit: 
            print('iteration limit exceeded!')
            more_data = False
    return inferred_velocities,inference_times,inferred_correlations

def z_mp_wrapper(i,j,signals,time,R,z,delta_ell,N,stepsize,correlation_threshold):
    j1,j2 = (j,j)
    i1,i2 = (i,i+delta_ell)
    sig1 = signals[i1,j1]
    R1,z1 = (R[i1,j1],z[i1,j1])
    sig2 = signals[i2,j2]
    R2,z2 = (R[i2,j2],z[i2,j2])
    velocities_one_channel,inference_times,correlations_one_channel = analyse_consecutive_clips_two_point(sig1,sig2,time,R1,R2,z1,z2,N,stepsize,correlation_threshold,mode='valid')
    if reverse_direction_check(i1,i2,z1,z2): velocities_one_channel = np.multiply(velocities_one_channel,-1.)
    return i,j,velocities_one_channel,inference_times,correlations_one_channel

def z_vel_scan_parallel(signals,time,j_range,i_range,R,z,N,stepsize,correlation_threshold='auto',delta_ell = 1,plot_bool=False):
    '''
    Scans field of view and performs velocimetry along the z (i) direction.
    Scan channel numbers can be specified in both i and j

    Arguments: (signals,time,j_range,i_range,R,z,N,correlation_threshold)
    Returns: inferred_velocities,inference_times

    Variables: 
    ----------
    signals: 3D numpy array [channel_i,channel_j,time]
        The signals to be analysed. Time assumed to be in seconds
    time: 1D numpy array [time]
        The times at which the signal datapoints were sampled.
        Time assumed to be in seconds.
    j_range,i_range: list or numpy array of integers
        The j/i channels to be scanned.
        j_range can include 0 to 7 including
        i_range can include 0 to 6 including
    R,z : 2D numpy array
        the R,z coordinates corresponding to the j,i channel numbers.
    N: integer 
        the length of the individual time-series to be analysed [number of frames]
    correlation_threshold: float
        threshold of correlation below which the inferred velocity will be ignored. [between 0 and 1]
    
    Keyword arguments:
    ------------------
    plot_bool: boolean
        should the CCF be plotted?
        WARNING! make sure you're not inside several nested loops :)
    delta_ell: integer
        what should the distance be between analysed channels?

    Returns:
    --------
    inferred_velocities: np array
        an [i_range,j_range,time] array containing inferred velocities
    inference_times: np array 
        contains the inference times of the velocities
    inference_correlations: np array 
        an [i_range,j_range,time] array containing correlation values of inferred velocities

    Notes:
    ------
    ::
    '''
    #run CCTDE in parallel
    nProcesses = len(i_range)*len(j_range)
    if correlation_threshold=='auto':
        correlation_threshold=calc_corr_threshold(N,0.01)
    with mp.Pool(processes= nProcesses) as pool:
        i_indices,j_indices,inferred_velocities,inference_times,inferred_correlations = zip(*pool.starmap(z_mp_wrapper, [(i,j,signals,time,R,z,delta_ell,N,stepsize,correlation_threshold) for i in i_range for j in j_range]))
    # convert to numpy arrays
    inferred_velocities = np.asarray(inferred_velocities)
    inferred_correlations = np.asarray(inferred_correlations)
    inference_times= np.asarray(inference_times)
    # reshape arrays
    inferred_velocities_reshaped = np.full((R.shape[0],R.shape[1],int(len(time)/stepsize)+1),np.nan)
    inferred_correlations_reshaped = np.full((R.shape[0],R.shape[1],int(len(time)/stepsize)+1),np.nan)
    for k,i in enumerate(i_indices):
        j = j_indices[k]
        inferred_velocities_reshaped[i,j,:] = inferred_velocities[k,:]
        inferred_correlations_reshaped[i,j,:] = inferred_correlations[k,:]
    inference_times_reshaped = inference_times[0,:]
    return inferred_velocities_reshaped,inference_times_reshaped,inferred_correlations_reshaped

def R_mp_wrapper(i,j,signals,time,R,z,delta_ell,N,stepsize,correlation_threshold):
    j1,j2 = (j,j+delta_ell)
    i1,i2 = (i,i)
    sig1 = signals[i1,j1]
    R1,z1 = (R[i1,j1],z[i1,j1])
    sig2 = signals[i2,j2]
    R2,z2 = (R[i2,j2],z[i2,j2])
    velocities_one_channel,inference_times,correlations_one_channel = analyse_consecutive_clips_two_point(sig1,sig2,time,R1,R2,z1,z2,N,stepsize,correlation_threshold,mode='valid')
    if reverse_direction_check(i1,i2,z1,z2): velocities_one_channel = np.multiply(velocities_one_channel,-1.)
    return i,j,velocities_one_channel,inference_times,correlations_one_channel

def R_vel_scan_parallel(signals,time,j_range,i_range,R,z,N,stepsize,correlation_threshold='auto',delta_ell = 1,plot_bool=False):
    '''
    Scans field of view and performs velocimetry along the R (j) direction.
    Scan channel numbers can be specified in both i and j

    Arguments: (signals,time,j_range,i_range,R,z,N,correlation_threshold)
    Returns: inferred_velocities,inference_times

    Variables: 
    ----------
    signals: 3D numpy array [channel_i,channel_j,time]
        The signals to be analysed. Time assumed to be in seconds
    time: 1D numpy array [time]
        The times at which the signal datapoints were sampled.
        Time assumed to be in seconds.
    j_range,i_range: list or numpy array of integers
        The j/i channels to be scanned.
        j_range can include 0 to 7 including
        i_range can include 0 to 6 including
    R,z : 2D numpy array
        the R,z coordinates corresponding to the j,i channel numbers.
    N: integer 
        the length of the individual time-series to be analysed [number of frames]
    correlation_threshold: float
        threshold of correlation below which the inferred velocity will be ignored. [between 0 and 1]
    
    Keyword arguments:
    ------------------
    plot_bool: boolean
        should the CCF be plotted?
        WARNING! make sure you're not inside several nested loops :)
    delta_ell: integer
        what should the distance be between analysed channels?

    Returns:
    --------
    inferred_velocities: np array
        an [i_range,j_range,time] array containing inferred velocities
    inference_times: np array 
        contains the inference times of the velocities
    inference_correlations: np array 
        an [i_range,j_range,time] array containing correlation values of inferred velocities

    Notes:
    ------
    ::
    '''
    #run CCTDE in parallel
    nProcesses = len(i_range)*len(j_range)
    if correlation_threshold=='auto':
        correlation_threshold=calc_corr_threshold(N,0.01)
    with mp.Pool(processes= nProcesses) as pool:
        i_indices,j_indices,inferred_velocities,inference_times,inferred_correlations = zip(*pool.starmap(R_mp_wrapper, [(i,j,signals,time,R,z,delta_ell,N,stepsize,correlation_threshold) for i in i_range for j in j_range]))
    # convert to numpy arrays
    inferred_velocities = np.asarray(inferred_velocities)
    inferred_correlations = np.asarray(inferred_correlations)
    inference_times= np.asarray(inference_times)
    # reshape arrays
    inferred_velocities_reshaped = np.full((R.shape[0],R.shape[1],int(len(time)/stepsize)+1),np.nan)
    inferred_correlations_reshaped = np.full((R.shape[0],R.shape[1],int(len(time)/stepsize)+1),np.nan)
    for k,i in enumerate(i_indices):
        j = j_indices[k]
        inferred_velocities_reshaped[i,j,:] = inferred_velocities[k,:]
        inferred_correlations_reshaped[i,j,:] = inferred_correlations[k,:]
    inference_times_reshaped = inference_times[0,:]
    return inferred_velocities_reshaped,inference_times_reshaped,inferred_correlations_reshaped


######################################################################################################################
######################################################################################################################
# helper line-cctde functions
######################################################################################################################
######################################################################################################################

def myline(x,m):
    #simple function of straight line
    #line is forced through 0,0
    return m*x

def estimate_gradient_with_ransac_through_origin(distances, lag_peaks, max_trials=100):
    # Check for NaN values and remove corresponding rows
    nan_mask = np.logical_or(np.isnan(distances), np.isnan(lag_peaks))
    distances = distances[~nan_mask]
    lag_peaks = lag_peaks[~nan_mask]
    # Reshape the arrays to 2D if they are 1D
    if len(distances.shape) == 1:
        distances = distances[:, np.newaxis]
    if len(lag_peaks.shape) == 1:
        lag_peaks = lag_peaks[:, np.newaxis]

    # Initialize RANSAC regressor with a linear model without an intercept
    ransac = RANSACRegressor(LinearRegression(fit_intercept=False), max_trials=max_trials)

    # Fit the regressor
    ransac.fit(distances, lag_peaks)

    # Get the inlier mask
    inliers = ransac.inlier_mask_
    
    # Extract inlier points
    inlier_distances = distances[inliers]
    inlier_lag_peaks = lag_peaks[inliers]

    # Fit the final model on the inliers
    final_model = RANSACRegressor(LinearRegression(fit_intercept=False))
    final_model.fit(inlier_distances, inlier_lag_peaks)

    # Get the estimated gradient
    gradient = final_model.estimator_.coef_[0]

    # Calculate the residuals for the inliers
    residuals = inlier_lag_peaks - final_model.predict(inlier_distances)
    # Calculate the standard error of the gradient
    n = len(inlier_distances)
    mse = np.sum(residuals ** 2) / (n - 2)  # Mean squared error
    gradient_error = np.sqrt(mse / np.sum((inlier_distances - np.mean(inlier_distances)) ** 2))
    
    # Visualization (optional)
    # plt.scatter(distances, lag_peaks, color='b', label='Data points')
    # plt.scatter(inlier_distances, inlier_lag_peaks, color='r', label='Inliers')
    # plt.plot(distances, final_model.predict(distances), color='orange', label='RANSAC Model')
    # plt.xlabel('Distances')
    # plt.ylabel('Lag Peaks')
    # plt.legend()
    # plt.show()
    return gradient, gradient_error

######################################################################################################################
######################################################################################################################
# core line-cctde functions
######################################################################################################################
######################################################################################################################

def infer_line_velocity(Linedata_segment,ref_j,zz,correlation_threshold,sampling_time,max_distance=None,exclude_edges=True,fit_method='ransac',plot_bool=False):
    """
    """
    # find all combination pairs of indices
    indices = range(Linedata_segment.shape[0])
    combinations = list(itertools.combinations(indices, 2))  # Generate all combinations of length 2 
    combinations= np.asarray(combinations) 
    index_exclusions = [] 
    for index,(iref,icomp) in enumerate(combinations):
        # optionally filter out edge indices
        if exclude_edges:
            if iref ==np.nanmin(indices) or icomp==np.nanmax(indices):
                index_exclusions.append(index)
        # filter out combinations with distances larger than max distance 
        if not max_distance==None:
            distance = (zz[iref,ref_j] - zz[icomp,ref_j])*100 #cm
            if distance > max_distance:
                index_exclusions.append(index)
    # remove combinations to be excluded
    index_exclusions = list(set(index_exclusions.copy()))
    combinations=np.delete(combinations,index_exclusions,axis=0)
    #initialise
    N = Linedata_segment.shape[1]//2
    lag_peaks=np.full(combinations.shape[0],np.nan)
    distances=np.full(combinations.shape[0],np.nan)
    # loop through all combinations
    for index,(iref,icomp) in enumerate(combinations):
        refsig = Linedata_segment[iref,N//2:N+N//2]
        compsig = Linedata_segment[icomp,:]
        # calculate CCF
        ccf,lags=calc_ccf(refsig,compsig,overlap_mode='valid',plot_bool=False)
        # check if ccf is empty
        if len(ccf) == 0:
            #leave lag_peak as nan and set correlation to 0
            correlation_max = 0.
        else:
            # find the peak of the cross-correlation function
            correlation_max = np.max(ccf)
            # correlation peak must exceed correlation threshold
            if correlation_max>correlation_threshold:
                #find time-delay at ccf peak
                peak_index=np.where(np.max(ccf)==ccf)[0][0]
                time_delay = lags[peak_index]
                #account for zero-time delay scenario
                if time_delay == 0.:
                    #leave lag_peak as nan
                    pass
                else:
                    #store time_delay and distance at which the CCF peaks
                    lag_peaks[index] = time_delay*sampling_time*1e6
                    distances[index] = -zz[iref,ref_j] + zz[icomp,ref_j]
            else:
                # leave lag_peak as nan if correlation threshold not reached
                pass
    # sort arrays for neatness
    sorted_indices = np.argsort(distances)
    distances = distances[sorted_indices].copy()
    lag_peaks = lag_peaks[sorted_indices].copy()
    # line fitting (multiple methods)
    if fit_method=='leastsq':
        mask = np.isfinite(lag_peaks)
        distances = distances[mask]
        lag_peaks = lag_peaks[mask]
        grad,grad_cov = curve_fit(myline,distances,lag_peaks)
        grad=grad[0]
        grad_err= grad_cov[0][0]
    elif fit_method=='ransac':
        grad,grad_err = estimate_gradient_with_ransac_through_origin(distances,lag_peaks)
        grad = grad[0]
    # calculate velocity in km/s
    velocity = 1./(grad/1000.)
    velocity_err = velocity*(grad_err/grad)
    #optional plotting
    if plot_bool:
        #fitted
        plt.plot(distances,distances*grad,ls='--',c='k',label='v={0:.2f}(+/-){1:.2f}km/s'.format(velocity,velocity_err))
        #raw
        plt.plot(distances,lag_peaks,'.',c='b')
        #formatting
        plt.title('line-CCTDE velocity fit, shot:46459, j=3 \n least squares method')
        plt.xlabel('distance [m]')
        plt.ylabel('time-delay [us]')
        plt.legend()
        plt.show()
    return velocity,velocity_err

def analyse_consecutive_clips_line(Linedata,times,N,stepsize,ref_j,zz,correlation_threshold,max_distance=None,exclude_edges=True,plot_bool=False,iterationlimit=1e9):
    """Analyze consecutive clips of density fluctuation data to infer z-velocities.

    This function analyzes consecutive clips of density fluctuations data (Linedata) to infer line velocities.
    It calculates the z-velocity by cross-correlating time-series slices and finding the peak correlation.

    Parameters:
        Linedata (numpy.ndarray): 2D array containing density fluctuations data.
        times (numpy.ndarray): 1D array of timestamps corresponding to the Linedata.
        N (int): Size of the window for cross-correlation.
        stepsize (int): Step size between consecutive clips of data.
        ref_j (int): Index of the reference location (column).
        zz (numpy.ndarray): 2D array of z-coordinates of the signals.
        correlation_threshold (float): Threshold value for the correlation coefficient to consider a valid peak.
        method (str, optional): Method for line fitting, can be 'ransac' (default), 'leastsq'.
        plot_bool (bool, optional): If True, plot the fitted line and raw data points. Defaults to False.
        iterationlimit (int, optional): Maximum number of iterations for the analysis loop. Defaults to 1e9.

    Returns:
        tuple: A tuple containing the inference times, inferred velocities, and their associated errors.

    Note:
        The function takes consecutive slices of the Linedata with a specified window size (N) and step size
        (stepsize). It performs cross-correlation on each slice to infer the line velocities. The function
        aborts if there is not enough data left in the time-series or if the iteration limit is exceeded.

    Example:
        times, velocities, velocity_errors = analyse_consecutive_clips_line(Linedata, times, 100, 10, 4, zz, 0.7)
        print("Inference Times:", times)
        print("Inferred Velocities:", velocities)
        print("Velocity Errors:", velocity_errors)
    """
    #initialise
    is_more_data=True
    i = 0
    sampling_time = np.nanmean(np.diff(times))
    nframes = len(Linedata[i,:])
    arr_length = int((nframes-N)/stepsize)+1
    inferred_velocities = np.full(arr_length,np.nan)
    inferred_velocity_errors = np.full(arr_length,np.nan)
    inference_times = np.full(arr_length,np.nan)
    #loop until there is no more data
    while is_more_data:
        #take slices of time-series
        Linedata_segment = Linedata[:,i:i+2*N]
        sliced_times = times[i:i+2*N]
        inference_time = np.nanmean(sliced_times)
        try:
            #cross-correlate ts slices and infer velocity
            velocity,velocity_err=infer_line_velocity(Linedata_segment,ref_j,zz,correlation_threshold,sampling_time,max_distance=max_distance,exclude_edges=exclude_edges,plot_bool=plot_bool)
        except:
            velocity = np.nan
            velocity_err = np.nan
        #store velocity in array
        inferred_velocities[int(i/stepsize)] = velocity
        inferred_velocity_errors[int(i/stepsize)] = velocity_err
        inference_times[int(i/stepsize)] = inference_time
        #move the current starting point
        i = i + stepsize
                # abort loop if there is not enough data left in the time-series
        if i+2*N >= nframes: is_more_data = False
        # abort if iterationlimit is exceeded
        if i/N > iterationlimit: 
            print('iteration limit exceeded!')
            is_more_data = False
    return inference_times,inferred_velocities,inferred_velocity_errors

def line_z_vel_scan(BESdata,BEStimes,shotn,RR,zz,N,stepsize,j_indices,max_distance=None,exclude_edges=True,correlation_threshold='auto',plot_bool=False):
    #initialise
    all_inferred_velocities = np.full((BESdata.shape[1],(len(BEStimes)-N)//stepsize+1),np.nan)
    all_inferred_velocities_err = np.full((BESdata.shape[1],(len(BEStimes)-N)//stepsize+1),np.nan)
    all_inference_times = np.full((len(BEStimes)-N+1)//stepsize,np.nan)
    # optionally calculate correlation threshold
    if correlation_threshold=='auto':
        correlation_threshold=calc_corr_threshold(N,0.01)
    elif not np.isnumeric(correlation_threshold):
        raise ValueError("correlation_threshold is not a number.")
    #loop over j indices
    for j in j_indices:
        BESdata_line = BESdata[:,j,:]
        inference_times,inferred_velocities,inferred_velocity_errors=analyse_consecutive_clips_line(BESdata_line,BEStimes,N,stepsize,j,zz,correlation_threshold,max_distance=max_distance,exclude_edges=exclude_edges,plot_bool=False)
        all_inferred_velocities[j,:] = inferred_velocities
        all_inferred_velocities_err[j,:] = inferred_velocity_errors
        all_inference_times = inference_times.copy()
    if plot_bool==True:
        for j in j_indices:
            plt.plot(all_inference_times*1000.,all_inferred_velocities[j,:],label='R={0:.2f}m'.format(RR[3,j]))
        plt.legend()
        plt.title('Overview of line-CCTDE inferred velocities \n shot:#{0},times {1:.2f}-{2:.2f}ms'.format(shotn,np.nanmin(all_inference_times)*1000.,np.nanmax(all_inference_times)*1000.))
        plt.ylabel('z-velocity [km/s]')
        plt.xlabel('time [ms]')
        plt.show()
    return all_inferred_velocities,all_inferred_velocities_err,all_inference_times

def process_j_index(j, BESdata, BEStimes, N, stepsize, zz, correlation_threshold, max_distance=None, exclude_edges=True):
    BESdata_line = BESdata[:, j, :]
    inference_times, inferred_velocities, inferred_velocity_errors = analyse_consecutive_clips_line(BESdata_line, BEStimes, N, stepsize, j, zz, correlation_threshold, max_distance=max_distance, exclude_edges=exclude_edges, plot_bool=False)
    return j, inferred_velocities, inferred_velocity_errors, inference_times

def line_z_vel_scan_parallel(BESdata, BEStimes, shotn, RR, zz, N, stepsize, j_indices, max_distance=None, exclude_edges=True, correlation_threshold='auto', plot_bool=False):
    all_inferred_velocities = np.full((BESdata.shape[1], (len(BEStimes) - N) // stepsize + 1), np.nan)
    all_inferred_velocities_err = np.full((BESdata.shape[1], (len(BEStimes) - N) // stepsize + 1), np.nan)
    all_inference_times = np.full((len(BEStimes) - N + 1) // stepsize, np.nan)
    
    if correlation_threshold == 'auto':
        correlation_threshold = calc_corr_threshold(N, 0.01)
    elif not np.isnumeric(correlation_threshold):
        raise ValueError("correlation_threshold is not a number.")
    
    num_processes = len(j_indices)
    pool = mp.Pool(processes=num_processes)
    
    results = []
    for j in j_indices:
        results.append(pool.apply_async(process_j_index, (j, BESdata, BEStimes, N, stepsize, zz, correlation_threshold, max_distance, exclude_edges)))
    
    pool.close()
    pool.join()
    
    for result in results:
        j, inferred_velocities, inferred_velocity_errors, inference_times = result.get()
        all_inferred_velocities[j, :] = inferred_velocities
        all_inferred_velocities_err[j, :] = inferred_velocity_errors
        all_inference_times = inference_times.copy()
    
    if plot_bool:
        for j in j_indices:
            plt.plot(all_inference_times * 1000., all_inferred_velocities[j, :], label='R={0:.2f}m'.format(RR[3, j]))
        plt.legend()
        plt.title('Overview of line-CCTDE inferred velocities \n shot:#{0},times {1:.2f}-{2:.2f}ms'.format(shotn, np.nanmin(all_inference_times) * 1000., np.nanmax(all_inference_times) * 1000.))
        plt.ylabel('z-velocity [km/s]')
        plt.xlabel('time [ms]')
        plt.show()
    
    return all_inferred_velocities, all_inferred_velocities_err, all_inference_times

######################################################################################################################
######################################################################################################################
# scripts in development or deprecated
######################################################################################################################
######################################################################################################################

# def R_vel_scan(signals,time,j_range,i_range,R,z,N,stepsize,correlation_threshold='auto',delta_ell = 1,plot_bool=False):
#     '''
#     Scans field of view and performs velocimetry along the R (j) direction.
#     Scan channel numbers can be specified in both i and j

#     Arguments: (signals,time,j_range,i_range,R,z,N,correlation_threshold)
#     Returns: inferred_velocities,inference_times

#     Variables: 
#     ----------
#     signals: 3D numpy array [channel_i,channel_j,time]
#         The signals to be analysed. Time assumed to be in seconds
#     time: 1D numpy array [time]
#         The times at which the signal datapoints were sampled.
#         Time assumed to be in seconds.
#     j_range,i_range: list or numpy array of integers
#         The j/i channels to be scanned.
#         j_range can include 0 to 7 including
#         i_range can include 0 to 6 including
#     R,z : 2D numpy array
#         the R,z coordinates corresponding to the j,i channel numbers.
#     N: integer 
#         the length of the individual time-series to be analysed [number of frames]
#     stepsize: integer
#         how many frames to step between velocity inferences
#     correlation_threshold: float
#         threshold of correlation below which the inferred velocity will be ignored. [between 0 and 1]
    
#     Keyword arguments:
#     ------------------
#     plot_bool: boolean
#         should the CCF be plotted?
#         WARNING! make sure you're not inside several nested loops :)
#     delta_ell: integer
#         what should the distance be between analysed channels?

#     Returns:
#     --------
#     inferred_velocities: np array
#         an [i_range,j_range,time] array containing inferred velocities
#     inference_times: np array 
#         contains the inference times of the velocities
#     inference_correlations: np array 
#         an [i_range,j_range,time] array containing correlation values of inferred velocities

#     Notes:
#     ------
#     ::
#     '''
#     inferred_velocities = np.full((8,8,int(len(time)/stepsize)+1),np.nan)
#     inferred_correlations = np.full((8,8,int(len(time)/stepsize)+1),np.nan)
#     if correlation_threshold=='auto':
#         correlation_threshold=calc_corr_threshold(N,0.01)
#     for j in j_range:
#         for i in i_range:
#             j1,j2 = (j,j+delta_ell)
#             i1,i2 = (i,i)
#             sig1 = signals[i1,j1]
#             R1,z1 = (R[i1,j1],z[i1,j1])
#             sig2 = signals[i2,j2]
#             R2,z2 = (R[i2,j2],z[i2,j2])
#             if plot_bool: print('i,j= {0},{1}'.format(i,j))
#             velocities_one_channel,inference_times,correlations_one_channel = analyse_consecutive_clips_two_point(sig1,sig2,time,R1,R2,z1,z2,N,stepsize,correlation_threshold,plot_bool=plot_bool)
#             if reverse_direction_check(i1,i2,z1,z2): velocities_one_channel = np.multiply(velocities_one_channel,-1.)
#             inferred_velocities[i,j,:] = velocities_one_channel
#             inferred_correlations[i,j,:] = correlations_one_channel
#     return inferred_velocities,inference_times,inferred_correlations

# def z_vel_scan(signals,time,j_range,i_range,R,z,N,stepsize,correlation_threshold='auto',delta_ell = 1,plot_bool=False):
#     '''
#     Scans field of view and performs velocimetry along the z (i) direction.
#     Scan channel numbers can be specified in both i and j
#     Currently only supports channel seperation of one.

#     Arguments: (signals,time,j_range,i_range,R,z,N,correlation_threshold)
#     Returns: inferred_velocities,inference_times

#     Variables: 
#     ----------
#     signals: 3D numpy array [channel_i,channel_j,time]
#         The signals to be analysed. Time assumed to be in seconds
#     time: 1D numpy array [time]
#         The times at which the signal datapoints were sampled.
#         Time assumed to be in seconds.
#     j_range,i_range: list or numpy array of integers
#         The j/i channels to be scanned.
#         j_range can include 0 to 7 including
#         i_range can include 0 to 6 including
#     R,z : 2D numpy array
#         the R,z coordinates corresponding to the j,i channel numbers.
#     N: integer 
#         the length of the individual time-series to be analysed [number of frames]
#     stepsize: integer
#         how many frames to step between velocity inferences
#     correlation_threshold: float
#         threshold of correlation below which the inferred velocity will be ignored. [between 0 and 1]
    
#     Keyword arguments:
#     ------------------
#     plot_bool: boolean
#         should the CCF be plotted?
#         WARNING! make sure you're not inside several nested loops :)
#     delta_ell: integer
#         what should the distance be between analysed channels?

#     Returns:
#     --------
#     inferred_velocities: np array
#         an [i_range,j_range,time] array containing inferred velocities
#     inference_times: np array 
#         contains the inference times of the velocities
#     inference_correlations: np array 
#         an [i_range,j_range,time] array containing correlation values of inferred velocities

#     Notes:
#     ------
#     ::
#     '''
#     inferred_velocities = np.full((8,8,int(len(time)/stepsize)+1),np.nan)
#     inferred_correlations = np.full((8,8,int(len(time)/stepsize)+1),np.nan)
#     if correlation_threshold=='auto':
#         correlation_threshold=calc_corr_threshold(N,0.01)
#     for j in j_range:
#         for i in i_range:
#             j1,j2 = (j,j)
#             i1,i2 = (i,i+delta_ell)
#             sig1 = signals[i1,j1]
#             R1,z1 = (R[i1,j1],z[i1,j1])
#             sig2 = signals[i2,j2]
#             R2,z2 = (R[i2,j2],z[i2,j2])
#             if plot_bool: print('i,j= {0},{1}'.format(i,j))
#             velocities_one_channel,inference_times,correlations_one_channel = analyse_consecutive_clips_two_point(sig1,sig2,time,R1,R2,z1,z2,N,stepsize,correlation_threshold,plot_bool=plot_bool)
#             if reverse_direction_check(i1,i2,z1,z2): velocities_one_channel = np.multiply(velocities_one_channel,-1.)
#             inferred_velocities[i,j,:] = velocities_one_channel
#             inferred_correlations[i,j,:] = correlations_one_channel
#     return inferred_velocities,inference_times,inferred_correlations

# def depr_infer_2D_velocity(time_series,ref_location,spatial_seperation,correlation_threshold):
#     '''

#     Deprecated code!

#     Infers velocity in 2D plane at a specified reference point.
#     Arguments: (time_series,ref_location,spatial_seperation,correlation_threshold)
#     Returns: velocities,correlations

#     Parameters
#     ----------
#     time_series: 3D numpy array [x-space, y-space, time]
#         a time_series of spatially resolved images.
#     ref_location: a pair of intergers [x_index,y_index]
#         the reference coordinates in the image where the velocity is to be inferred.
#     spatial_seperation : integer
#         the spatial seperation between the two time-series of the ccf. Given in number of spatial channels.
#     correlation_threshold : float between 0 and 1
#         defines the minimum correlation required for velocity inference. If below threshold then velocity defaults to np.nan.

#     Returns
#     -------
#     velocities: a pair of floats [x_velocity,y_velocity]
#         inferred velocities at reference location.
#     correlations: a pair of floats [x_correlation,y_correlation]
#         peak amplitude of the correlation function used for velocity inference.

#     Notes
#     -----
#     :: In the current implementation, the selection of the locations is a bit illogical.
#        The reference location should be centered between the locations of the timeseries.
#        Should be changed before application to experiment.
#     '''
#     #select the timeseries
#     i,j = ref_location
#     ref = time_series[i,j,:]
#     x_ts = time_series[i+spatial_seperation,j,:]
#     y_ts = time_series[i,j+spatial_seperation,:]
#     #calculate velocities
#     velocities = np.empty(2)
#     correllations = np.empty(2)
#     x_ccf,x_lags = calc_ccf(ref,x_ts)
#     x_vel,x_correlation=infer_two_point_velocity(x_ccf,x_lags,spatial_seperation,correlation_threshold)
#     y_ccf,y_lags = calc_ccf(ref,y_ts)
#     y_vel,y_correlation=infer_two_point_velocity(y_ccf,y_lags,spatial_seperation,correlation_threshold)
#     velocities = (x_vel,y_vel)
#     correlations = (x_correlation,y_correlation)
#     return velocities,correlations

