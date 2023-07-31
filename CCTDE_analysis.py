import numpy as np 
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
print('Note: RuntimeWarnings are currently ignored in CCTDE_analysis')

######################################################################################################################
######################################################################################################################
# basic CCTDE analysis functions
######################################################################################################################
######################################################################################################################

def calc_velocity_mean(velocities,z_average=True,t_average=True):
    '''
    Calculates the mean velocity
    Arguments: (velocities,z_average=True,t_average=True)
    Returns: mean_vels

    Variables:
    ----------
    velocities: 3D np.array of floats
        contains inferred velocities in shape [z,R,time]

    Keyword arguments:
    ------------------
    z_average: boolean
        average over z-axis?
    t_average: boolean
        average over time-axis?

    Returns:
    --------

    '''
    if z_average and t_average:
        mean_vels=np.nanmean(velocities,axis = (0,2))
    elif z_average and not t_average:
        mean_vels=np.nanmean(velocities,axis = 0)
    return mean_vels

def calc_velocity_median(velocities,z_average=True,t_average=True):
    if z_average and t_average:
        median_vels=np.nanmedian(velocities,axis = (0,2))
    elif z_average and not t_average:
        median_vels=np.nanmedian(velocities,axis = 0)
    return median_vels

def calc_velocity_reciprocal_mean(velocities,z_average=True,t_average=True):
    inv_velocities = 1./velocities
    if z_average and t_average:
        inv_mean_vels=np.nanmean(inv_velocities,axis = (0,2))
    elif z_average and not t_average:
        inv_mean_vels=np.nanmean(inv_velocities,axis = 0)
    mean_vels = 1./inv_mean_vels
    return mean_vels

def calc_velocity_reciprocal_median(velocities,z_average=True,t_average=True):
    inv_velocities = 1./velocities
    if z_average and t_average:
        inv_median_vels=np.nanmedian(inv_velocities,axis = (0,2))
    elif z_average and not t_average:
        inv_median_vels=np.nanmedian(inv_velocities,axis = 0)
    median_vels = 1./inv_median_vels
    return median_vels

def calc_velocity_percentile_limits(velocities,p,z_average=True,t_average=True):
    if z_average and t_average:
        # combine axes 0 and 2 (i and time) of inferred_velocities
        flattened_vels = velocities.swapaxes(1,2).reshape(-1,velocities.shape[1]) # [i and time,j]
        #initialise storage arrays
        upper_velocities = np.full(flattened_vels.shape[1],np.nan)
        lower_velocities = np.full(flattened_vels.shape[1],np.nan)
        for j in range(flattened_vels.shape[1]):
            # calculate upper and lower percentiles
            upper_velocities[j] = np.nanpercentile(flattened_vels[:,j],100.*(1.+p)/2.,interpolation='midpoint')
            lower_velocities[j] = np.nanpercentile(flattened_vels[:,j],100.*(1.-p)/2.,interpolation='midpoint')
    elif z_average and not t_average:
        #initialise storage arrays
        upper_velocities = np.full((velocities.shape[1],velocities.shape[2]),np.nan)
        lower_velocities = np.full((velocities.shape[1],velocities.shape[2]),np.nan)
        for j in range(velocities.shape[1]):
            for t in range(velocities.shape[2]):
                # calculate upper and lower percentiles
                upper_velocities[j,t] = np.nanpercentile(velocities[:,j,t],100.*(1.+p)/2.,interpolation='midpoint')
                lower_velocities[j,t] = np.nanpercentile(velocities[:,j,t],100.*(1.-p)/2.,interpolation='midpoint')   
    return upper_velocities,lower_velocities


def reciprocal_velocity_averaging(velocities,time_average = True,z_average = False):
    '''
    Averages velocities array in reciprocal space
    Arguments: (velocities,time_average = True,z_average = False)
    Returns: avg_velocities,pos_stdev,neg_stdev,median_velocities,pos_mads,neg_mads

    Variables:
    ----------
    velocities: 3D numpy array [i,j,time]
        containins the inferred CCTDE velocities
    correlations: 3D numpy array [i,j,time]
        contains the correlation values corresponding to the inferred velocities
    
    Keyword arguments: 
    ------------------
    time_average: boolean
        average over time axis?
        Defaults to True
    z_average: boolean
        average over z spatial axis?
        Defaults to False

    Returns:
    --------
    avg_velocities: numpy array
        average velocities.
        shape depends on what axes have been averaged over.
    pos/neg_stdev: numpy array
        'standard' deviation of velocities above/below mean
        shape depends on what axes have been averaged over.
    median_velocities: numpy array
        median velocities
        shape depends on what axes have been averaged over.
    pos/neg_mads: numpy array
        median absolute deviations above/below median
        shape depends on what axes have been averaged over.

    Notes:
    ------
    :: Expects specifically the velocity array from vel_scan() in CCTDE_core.py
    '''
    # need to take the reciprocal to get proper averaging of the time-lags from CCTDE.
    # straight averaging would give skewed results.
    reciprocal_velocities = np.divide(1.,velocities)
    if time_average and z_average: 
        #take time and z-averages
        reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = (0,2))
        reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = (0,2))
        # calculate deviations above and below the reciprocal mean
        reciprocal_devs_from_mean = reciprocal_velocities - reciprocal_avg_velocities[np.newaxis,...,np.newaxis]
        reciprocal_positive_devs_from_mean = np.where(reciprocal_devs_from_mean>0.,reciprocal_devs_from_mean,np.nan)
        reciprocal_negative_devs_from_mean = np.where(reciprocal_devs_from_mean<0.,reciprocal_devs_from_mean,np.nan)
        # length of arrays above and below reciprocal mean 
        N_pos = np.count_nonzero(~np.isnan(reciprocal_positive_devs_from_mean))
        N_neg = np.count_nonzero(~np.isnan(reciprocal_negative_devs_from_mean))
        # calculate reciprocal stdevs
        reciprocal_pos_stdev = np.sqrt(np.nansum(reciprocal_positive_devs_from_mean**2)/N_pos)
        reciprocal_neg_stdev = np.sqrt(np.nansum(reciprocal_negative_devs_from_mean**2)/N_neg)
        # calculate reciprocal median absolute deviations
        reciprocal_median_devs = reciprocal_velocities - reciprocal_median_velocities[np.newaxis,...,np.newaxis]
        reciprocal_positive_deviation_from_median = np.abs(np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan))
        reciprocal_negative_deviation_from_median = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
        reciprocal_pos_mads = np.nanmedian(reciprocal_positive_deviation_from_median,axis= (0,2))
        reciprocal_neg_mads = np.nanmedian(reciprocal_negative_deviation_from_median,axis= (0,2))
    elif time_average and not z_average: 
        #take time average
        reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = 2)
        reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = 2)
        # calculate deviations above and below the reciprocal mean
        reciprocal_devs_from_mean = reciprocal_velocities - reciprocal_avg_velocities[...,np.newaxis]
        reciprocal_positive_devs_from_mean = np.where(reciprocal_devs_from_mean>0.,reciprocal_devs_from_mean,np.nan)
        reciprocal_negative_devs_from_mean = np.where(reciprocal_devs_from_mean<0.,reciprocal_devs_from_mean,np.nan)
        # length of arrays above and below reciprocal mean 
        N_pos = np.count_nonzero(~np.isnan(reciprocal_positive_devs_from_mean))
        N_neg = np.count_nonzero(~np.isnan(reciprocal_negative_devs_from_mean))
        # calculate reciprocal stdevs
        reciprocal_pos_stdev = np.sqrt(np.nansum(reciprocal_positive_devs_from_mean**2)/N_pos)
        reciprocal_neg_stdev = np.sqrt(np.nansum(reciprocal_negative_devs_from_mean**2)/N_neg)
        # calculate reciprocal median absolute deviations
        reciprocal_median_devs = reciprocal_velocities - reciprocal_median_velocities[...,np.newaxis]
        reciprocal_positive_deviation_from_median = np.abs(np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan))
        reciprocal_negative_deviation_from_median = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
        reciprocal_pos_mads = np.nanmedian(reciprocal_positive_deviation_from_median,axis= 2)
        reciprocal_neg_mads = np.nanmedian(reciprocal_negative_deviation_from_median,axis= 2)
    # convert from reciprocal space back to normal space
    avg_velocities = np.divide(1.,reciprocal_avg_velocities)
    median_velocities = np.divide(1.,reciprocal_median_velocities)
    # apply correct scaling to the errors
    # note that signs are opposite because of reciprocal
    pos_stdev = avg_velocities * (reciprocal_neg_stdev/reciprocal_avg_velocities)
    neg_stdev = avg_velocities * (reciprocal_pos_stdev/reciprocal_avg_velocities)
    pos_mads = median_velocities * (reciprocal_neg_mads/reciprocal_median_velocities)
    neg_mads = median_velocities * (reciprocal_pos_mads/reciprocal_median_velocities)
    return avg_velocities,pos_stdev,neg_stdev,median_velocities,pos_mads,neg_mads

def nonreciprocal_velocity_averaging(velocities,time_average = True,z_average = False):
    '''
    Averages velocities array in nonreciprocal space
    Arguments: (velocities,time_average = True,z_average = False)
    Returns: avg_velocities,pos_stdev,neg_stdev,median_velocities,pos_mads,neg_mads

    Variables:
    ----------
    velocities: 3D numpy array [i,j,time]
        containins the inferred CCTDE velocities
    correlations: 3D numpy array [i,j,time]
        contains the correlation values corresponding to the inferred velocities
    
    Keyword arguments: 
    ------------------
    time_average: boolean
        average over time axis?
        Defaults to True
    z_average: boolean
        average over z spatial axis?
        Defaults to False

    Returns:
    --------
    avg_velocities: numpy array
        average velocities.
        shape depends on what axes have been averaged over.
    pos/neg_stdev: numpy array
        'standard' deviation of velocities above/below mean
        shape depends on what axes have been averaged over.
    median_velocities: numpy array
        median velocities
        shape depends on what axes have been averaged over.
    pos/neg_mads: numpy array
        median absolute deviations above/below median
        shape depends on what axes have been averaged over.

    Notes:
    ------
    :: 
    '''
    if time_average and z_average: 
        #take time and z-averages
        avg_velocities = np.nanmean(velocities,axis = (0,2))
        median_velocities = np.nanmedian(velocities,axis = (0,2))
        # calculate deviations above and below the reciprocal mean
        devs_from_mean = velocities - avg_velocities[np.newaxis,...,np.newaxis]
        positive_devs_from_mean = np.where(devs_from_mean>0.,devs_from_mean,np.nan)
        negative_devs_from_mean = np.where(devs_from_mean<0.,devs_from_mean,np.nan)
        # length of arrays above and below reciprocal mean 
        N_pos = np.count_nonzero(~np.isnan(positive_devs_from_mean))
        N_neg = np.count_nonzero(~np.isnan(negative_devs_from_mean))
        # calculate reciprocal stdevs
        pos_stdev = np.sqrt(np.nansum(positive_devs_from_mean**2)/N_pos)
        neg_stdev = np.sqrt(np.nansum(negative_devs_from_mean**2)/N_neg)
        # calculate reciprocal median absolute deviations
        median_devs = velocities - median_velocities[np.newaxis,...,np.newaxis]
        positive_deviation_from_median = np.where(median_devs>0.,median_devs,np.nan)
        negative_deviation_from_median = np.where(median_devs<0.,median_devs,np.nan)
        pos_mads = np.abs(np.nanmedian(positive_deviation_from_median,axis= (0,2)))
        neg_mads = np.abs(np.nanmedian(negative_deviation_from_median,axis= (0,2)))
    elif time_average and not z_average: 
        #take time average
        avg_velocities = np.nanmean(velocities,axis = 2)
        median_velocities = np.nanmedian(velocities,axis = 2)
        # calculate deviations above and below the reciprocal mean
        devs_from_mean = velocities - avg_velocities[...,np.newaxis]
        positive_devs_from_mean = np.where(devs_from_mean>0.,devs_from_mean,np.nan)
        negative_devs_from_mean = np.where(devs_from_mean<0.,devs_from_mean,np.nan)
        # length of arrays above and below reciprocal mean 
        N_pos = np.count_nonzero(~np.isnan(positive_devs_from_mean))
        N_neg = np.count_nonzero(~np.isnan(negative_devs_from_mean))
        # calculate reciprocal stdevs
        pos_stdev = np.sqrt(np.nansum(positive_devs_from_mean**2)/N_pos)
        neg_stdev = np.sqrt(np.nansum(negative_devs_from_mean**2)/N_neg)
        # calculate reciprocal median absolute deviations
        median_devs = velocities - median_velocities[...,np.newaxis]
        positive_deviation_from_median = np.where(median_devs>0.,median_devs,np.nan)
        negative_deviation_from_median = np.where(median_devs<0.,median_devs,np.nan)
        pos_mads = np.abs(np.nanmedian(positive_deviation_from_median,axis= 2))
        neg_mads = np.abs(np.nanmedian(negative_deviation_from_median,axis= 2))
    return avg_velocities,pos_stdev,neg_stdev,median_velocities,pos_mads,neg_mads

def remove_reciprocal_outliers(inferred_velocities,threshold):
    '''
    Removes outliers in reciprocal space. 'Outlier' defined as being more than IQR*threshold away from the median.
    '''
    reciprocal_velocities = 1./inferred_velocities
    # combine axes 0 and 2 (i and time) of inferred_velocities
    flattened_vels = reciprocal_velocities.swapaxes(1,2).reshape(-1,reciprocal_velocities.shape[1]) # [i and time,j]
    # initialise loop over j
    median_vels = np.nanmedian(reciprocal_velocities,axis=(0,2))
    cleaned_vels = np.full(reciprocal_velocities.shape,np.nan)
    for j in range(flattened_vels.shape[1]):
        # calculate inter-quartile range over i and time dimensions
        Q1 = np.nanpercentile(flattened_vels[:,j],25,interpolation='midpoint')
        Q3 = np.nanpercentile(flattened_vels[:,j],75,interpolation='midpoint')
        iqr = Q3 - Q1
        # extract only non-outliers into cleaned_vels
        cleaned_vels[:,j,:]=np.where(np.abs(reciprocal_velocities[:,j,:]-median_vels[j])< threshold*iqr,reciprocal_velocities[:,j,:],np.nan)
    return 1./cleaned_vels

def remove_nonreciprocal_outliers(inferred_velocities,threshold):
    '''
    Removes outliers in reciprocal space. 'Outlier' defined as being more than IQR*threshold away from the median.
    '''
    # combine axes 0 and 2 (i and time) of inferred_velocities
    flattened_vels = inferred_velocities.swapaxes(1,2).reshape(-1,inferred_velocities.shape[1]) # [i and time,j]
    # initialise loop over j
    median_vels = np.nanmedian(inferred_velocities,axis=(0,2))
    cleaned_vels = np.full(inferred_velocities.shape,np.nan)
    for j in range(flattened_vels.shape[1]):
        # calculate inter-quartile range over i and time dimensions
        Q1 = np.nanpercentile(flattened_vels[:,j],25,interpolation='midpoint')
        Q3 = np.nanpercentile(flattened_vels[:,j],75,interpolation='midpoint')
        iqr = Q3 - Q1
        # extract only non-outliers into cleaned_vels
        cleaned_vels[:,j,:]=np.where(np.abs(inferred_velocities[:,j,:]-median_vels[j])< threshold*iqr,inferred_velocities[:,j,:],np.nan)
    return cleaned_vels



def clean_velocities(inferred_velocities,threshold,type='reciprocal'):
    '''
    Sets outliers to NaN of inferred velocities array. Currently only outliers in reciprocal space filtered.
    Arguments: inferred_velocities,threshold
    Returns: cleaned_velocities

    Variables:
    ----------
    inferred_velocities: 3D numpy array, floats
        a [i,j,time] array containing velocity inferences.
    threshold: float
        Outliers defined as more than IQR*threshold away from median. Outliers found in reciprocal space.

    Keyword arguments:
    ------------------
    type: string
        'reciprocal' - remove outliers in reciprocal space
        'nonreciprocal' - remove outliers in nonreciprocal space
        'both' - remove in both spaces

    Returns:
    --------
    cleaned_velocities: 3D numpy array, floats
        same shape as inferred_velocities. Contains velocities with reciprocal outliers removed.
    '''
    if type=='nonreciprocal':
        cleaned_velocities = remove_nonreciprocal_outliers(inferred_velocities,threshold)
    elif type=='reciprocal':
        cleaned_velocities = remove_reciprocal_outliers(inferred_velocities,threshold)
    elif type=='both':
        cleaned_velocities = remove_nonreciprocal_outliers(inferred_velocities,threshold)
        cleaned_velocities = remove_reciprocal_outliers(cleaned_velocities,threshold)
    else:
        print('Unrecognised type passed. Stop.')
        print(1/0)
    return cleaned_velocities


def acceleration_filter(velocities,inference_times,max_acceleration,plot_bool = False):
    '''
    Filters out accelerations above a given threshold.
    Arguments: (velocities,inference_times,max_acceleration)
    Returns: filtered_velocities,accelerations

    Variables:
    ----------
    velocities: 1D numpy array, floats
        array containing velocities
    inference_times: 1D numpy array, floats
        array containing inference times of the velocities
    max_acceleration: float
        accelerations above this value will get filtered out

    Returns:
    --------
    filtered_velocities: 1D numpy array, floats
        array containing filtered velocities
    accelerations: 1D numpy array, floats
        array containting accelerations

    Notes:
    ------
    :: Calculates acceleration by taking the difference between 
        a velocity and the most recent non-nan velocity, 
        then dividing by time difference.
    '''
    iterationlimit = 10000
    #initialise accelerations array 
    accelerations = np.full(len(velocities),np.nan)
    filtered_velocities = np.full(len(velocities),np.nan)
    # calculate the sampling time
    dt = np.mean(np.diff(inference_times))
    # loop from second to last index of velocities
    for t in range(1,len(velocities)):
        # try inceasingly further timeshifts before t
        for tshift in np.arange(1,iterationlimit):
            # break if shift is larger than t
            if t-tshift<0: break
            # if the previous velocity is nan, then pass to next loop
            if np.isnan(velocities[t-tshift]):
                pass
            # if the current velocity is nan, then break.
            elif np.isnan(velocities[t]):
                break
            # if both velocities are not nan, calculate acceleration
            elif not np.isnan(velocities[t-tshift]):
                accelerations[t] = (velocities[t] - velocities[t-tshift])/(dt*tshift)
            break
        # if acceleration is above threshold, set velocity to nan.
        if np.abs(accelerations[t]) > max_acceleration:
            filtered_velocities[t] = np.nan
        else:
            filtered_velocities[t] = velocities[t]
    if plot_bool:
        plt.figure(figsize=(12,10))
        plt.plot(inference_times,velocities,marker='.',ls='',label='raw velocities')
        plt.plot(inference_times,filtered_velocities,label='filtered velocites')
        plt.xlabel('inference times [s]')
        plt.ylabel('inferred velocities [km/s]')
        plt.legend()
    return filtered_velocities,accelerations

###########################################################################################
###########################################################################################
# main CCTDE plotting functions
###########################################################################################
###########################################################################################

def plot_vel_time_one_location(velocities,times, i, j, shotn, N,vlim = 'all',tlim = 'all',plot_average='median', linest = '-'):
    #pick velocity timeseries at one location
    plotting_velocities = velocities[i,j,:]
    #plot velocities against time
    plt.figure(figsize=(10,8))
    plt.plot(times,plotting_velocities,ls=linest,marker='.')
    plt.xlabel('Time [s]')
    plt.ylabel('Inferred velocity [km/s]')
    plt.title('Shotno:{0}, N: {3} \n Velocity from channel {1} to {2}'.format(shotn,(i,j),(i+1,j),N))
    #set plotting limits if desired
    if vlim == 'all':
        pass
    else:
        minv,maxv = vlim
        plt.ylim(minv,maxv)
    if tlim == 'all':
        pass
    else:
        mint,maxt = tlim
        plt.xlim(mint,maxt)
    if plot_average == 'median':
        median = np.nanmedian(plotting_velocities)
        plt.hlines(median,np.nanmin(times),np.nanmax(times),linestyles='--',label= 'median: {0:.3f}km/s'.format(median))
        plt.legend()
    elif plot_average == 'mean':
        mean = np.nanmean(plotting_velocities)
        plt.hlines(mean,np.nanmin(times),np.nanmax(times),linestyles='--',label= 'mean: {0:.3f}km/s'.format(mean))
        plt.legend()
    elif plot_average == 'both':
        median = np.nanmedian(plotting_velocities)
        plt.hlines(median,np.nanmin(times),np.nanmax(times),linestyles='--',label= 'median: {0:.3f}km/s'.format(median))
        mean = np.nanmean(plotting_velocities)
        plt.hlines(mean,np.nanmin(times),np.nanmax(times),linestyles='--',label= 'mean: {0:.3f}km/s'.format(mean))
        plt.legend()
    elif plot_average == 'none':
        pass
    plt.show()
    return

def plot_vel_R_avg_time(velocities,times,i_range,shotn,N,R,z,vlim = 'all',Rlim = 'all',metric='mean'):
    averaged_velocities,pos_stdevs,neg_stdevs,median_velocities,pos_mads,neg_mads = reciprocal_velocity_averaging(velocities,time_average=True,z_average=False)
    plt.figure(figsize=(10,8))
    if metric == 'mean':
        for i in i_range:
            plt.plot(R[i,:],averaged_velocities[i,:],marker='.',label = 'z:{:.2f}m'.format(z[i,0]))
            plt.fill_between(R[i,:],averaged_velocities[i,:]-neg_stdevs[i,:],averaged_velocities[i,:]+pos_stdevs[i,:],alpha=0.2)
            plt.ylabel('Mean inferred velocity [km/s]')
    elif metric == 'median':
        for i in i_range:
            plt.plot(R[i,:],median_velocities[i,:],marker='.',label = 'z:{:.2f}m'.format(z[i,0]))
            plt.fill_between(R[i,:],median_velocities[i,:]-neg_mads[i,:],median_velocities[i,:]+pos_mads[i,:],alpha=0.2)
            plt.ylabel('Median inferred velocity [km/s]')
    elif metric == 'both':
        for i in i_range:
            plt.plot(R[i,:],averaged_velocities[i,:],marker='.',label = 'mean, z:{:.2f}m'.format(z[i,0]))
            plt.fill_between(R[i,:],averaged_velocities[i,:]-neg_stdevs[i,:],averaged_velocities[i,:]+pos_stdevs[i,:],alpha=0.2)
            plt.plot(R[i,:],median_velocities[i,:],marker='.',label = 'median, z:{:.2f}m'.format(z[i,0]))
            plt.fill_between(R[i,:],median_velocities[i,:]-neg_mads[i,:],median_velocities[i,:]+pos_mads[i,:],alpha=0.2)
            plt.ylabel('Inferred velocity [km/s]')
    plt.xlabel('Major radius [m]')
    plt.title('Shotno:{0}, N: {1} \n Time-averaged: [{2:.2f}-{3:.2f}]s'.format(shotn,N,np.nanmin(times),np.nanmax(times)))
    #set plotting limits if desired
    if vlim == 'all':
        pass
    else:
        minv,maxv = vlim
        plt.ylim(minv,maxv)
    if Rlim == 'all':
        pass
    else:
        minR,maxR = Rlim
        plt.xlim(minR,maxR)
    plt.legend()
    plt.grid()
    return

def plot_vel_R_avg_time_avg_z(velocities,times,shotn,N,R,vlim = 'all',Rlim = 'all',metric = 'mean'):
    averaged_velocities,pos_stdevs,neg_stdevs,median_velocities,pos_mads,neg_mads = reciprocal_velocity_averaging(velocities,time_average=True,z_average=True)
    plt.figure(figsize=(10,8))
    if metric == 'mean':
        plt.plot(R[0,:],averaged_velocities,marker='.',label = 'mean',color='b')
        plt.fill_between(R[0,:],averaged_velocities-neg_stdevs,averaged_velocities+pos_stdevs,alpha=0.2,color='b')
    elif metric == 'median':
        plt.plot(R[0,:],median_velocities,marker='.',label = 'median',color = 'orange')
        plt.fill_between(R[0,:],median_velocities-neg_mads,median_velocities+pos_mads,alpha=0.2,color = 'orange')
    elif metric == 'both':
        plt.plot(R[0,:],averaged_velocities,marker='.',label = 'mean',color = 'b')
        plt.fill_between(R[0,:],averaged_velocities-neg_stdevs,averaged_velocities+pos_stdevs,alpha=0.2,color = 'b')
        plt.plot(R[0,:],median_velocities,marker='.',label = 'median',color = 'orange')
        plt.fill_between(R[0,:],median_velocities-neg_mads,median_velocities+pos_mads,alpha=0.2,color = 'orange')
    plt.xlabel('Major radius [m]')
    plt.ylabel('Inferred velocity [km/s]')
    plt.title('Shotno:{0}, N: {1} \n z-averaged, Time-averaged: [{2:.3f}-{3:.3f}]s'.format(shotn,N,np.nanmin(times),np.nanmax(times)))
    plt.legend()
    plt.grid()
    #set plotting limits if desired
    if vlim == 'all':
        pass
    else:
        minv,maxv = vlim
        plt.ylim(minv,maxv)
    if Rlim == 'all':
        pass
    else:
        minR,maxR = Rlim
        plt.xlim(minR,maxR)
    plt.legend()
    return

def rolling_mean(arr,N,stepsize,space='direct'):
    if space=='direct':
        mean_arr=rolling_mean_singlearray(arr,N,stepsize)
        return mean_arr
    elif space=='reciprocal':
        # get maximum velocity
        max_vel = np.nanmax(arr)
        reciprocal_max_vel = 1./max_vel
        #flip the array 
        arr = 1./arr.copy()
        # take mean of reciprocal velocities
        mean_arr=rolling_mean_singlearray(arr,N,stepsize)
        mean_arr=np.where(np.abs(mean_arr)>reciprocal_max_vel,mean_arr,np.nan)
        # flip the mean velocities to direct space
        mean_arr = 1./mean_arr.copy()
        return mean_arr
    
def rolling_mean_singlearray(arr,N,stepsize):
    # initialise an unpacked array 
    unpacked_arr = np.full((len(arr),N),np.nan)
    # loop over array indices excluding
    for i in np.arange(N//2,len(arr)-1-N//2,stepsize):
        # set start and end indices of array slice. Includes edge cases. 
        start_index = int(i-N/2)
        end_index = int(i+N/2)
        # add array slice to the unpacked array
        unpacked_arr[i,:] = arr[start_index:end_index]
    # average over the segments, giving the rolling average
    mean_arr = np.nanmean(unpacked_arr,axis=1)
    return mean_arr
    
def rolling_median(arr,N,stepsize,space='direct'):
    if space=='direct':
        median_arr=rolling_median_singlearray(arr,N,stepsize)
        return median_arr
    elif space=='reciprocal':
        # get maximum velocity
        max_vel = np.nanmax(arr)
        reciprocal_max_vel = 1./max_vel
        #flip the array 
        arr = 1./arr.copy()
        # take median of reciprocal velocities
        median_arr=rolling_median_singlearray(arr,N,stepsize)
        median_arr=np.where(np.abs(median_arr)>reciprocal_max_vel,median_arr,np.nan)
        # flip the median velocities to direct space
        median_arr = 1./median_arr.copy()
        return median_arr

def rolling_median_singlearray(arr,N,stepsize):
    # initialise an unpacked array 
    unpacked_arr = np.full((len(arr),N),np.nan)
    # loop over array indices excluding
    for i in np.arange(N//2,len(arr)-1-N//2,stepsize):
        # set start and end indices of array slice. Includes edge cases. 
        start_index = int(i-N/2)
        end_index = int(i+N/2)
        # add array slice to the unpacked array
        unpacked_arr[i,:] = arr[start_index:end_index]
    # average over the segments, giving the rolling average
    median_arr = np.nanmedian(unpacked_arr,axis=1)
    return median_arr

###########################################################################################
###########################################################################################
# deprecated functions 
###########################################################################################
###########################################################################################

# def velocity_averaging(velocities,correlations,time_average = True,z_average = False,weighted=True):
#     '''
#     Averages inferred_velocitied array from vel_scan functions in CCTDE_core.py
#     Arguments: (velocities,correlations,time_average = True,z_average = False,weighted=True)
#     Returns: avg_velocities,pos_stdev,neg_stdev,median_velocities,pos_mads,neg_mads

#     Variables:
#     ----------
#     velocities: 3D numpy array [i,j,time]
#         containins the inferred CCTDE velocities
#     correlations: 3D numpy array [i,j,time]
#         contains the correlation values corresponding to the inferred velocities
    
#     Keyword arguments: 
#     ------------------
#     time_average: boolean
#         average over time axis?
#         Defaults to True
#     z_average: boolean
#         average over z spatial axis?
#         Defaults to False
#     weighted: boolean
#         weight the averaging according to supplied correlations
#         Defaults to True

#     Returns:
#     --------
#     avg_velocities: numpy array
#         average velocities.
#         shape depends on what axes have been averaged over.
#     stdevs: numpy array
#         velocity standard deviation
#         shape depends on what axes have been averaged over.
#     median_velocities: numpy array
#         median velocities
#         shape depends on what axes have been averaged over.
#     mads: numpy array
#         median absolute deviations
#         shape depends on what axes have been averaged over.

#     Notes:
#     ------
#     :: 
#     '''
#     # need to take the reciprocal to get proper averaging of the time-lags from CCTDE.
#     # straight averaging would give skewed results.
#     reciprocal_velocities = np.divide(1.,velocities)
#     # calculate weighted averages
#     if weighted: 
#         if time_average and z_average: #checked
#             # mask array where there are nans
#             masked = np.ma.MaskedArray(reciprocal_velocities,mask = np.isnan(reciprocal_velocities))
#             # take average velocity
#             reciprocal_avg_velocities = np.ma.average(masked,axis = (0,2),weights=correlations)
#             # calculate deviations above and below the mean
#             reciprocal_mean_devs = masked - reciprocal_avg_velocities[np.newaxis,...,np.newaxis]
#             rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
#             rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
#             N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
#             N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
#             reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
#             reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
#             # take median velocity
#             reciprocal_median_velocities = np.ma.median(masked,axis = (0,2))
#             # calsulate median absolute deviations
#             reciprocal_median_devs = masked - reciprocal_median_velocities[np.newaxis,...,np.newaxis]
#             abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
#             abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
#             reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= (0,2))
#             reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= (0,2))
#         if time_average and not z_average: #checked
#             masked = np.ma.MaskedArray(reciprocal_velocities,mask = np.isnan(reciprocal_velocities))
#             reciprocal_avg_velocities = np.ma.average(masked,axis = 2,weights=correlations)
#             # calculate deviations above and below the mean
#             reciprocal_mean_devs = masked - reciprocal_avg_velocities[...,np.newaxis]
#             rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
#             rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
#             N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
#             N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
#             reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
#             reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
#             # take median velocity
#             reciprocal_median_velocities = np.ma.median(masked,axis = 2)
#             # calsulate median absolute deviations
#             reciprocal_median_devs = masked - reciprocal_median_velocities[...,np.newaxis]
#             abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
#             abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
#             reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= 2)
#             reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= 2)
#         if z_average and not time_average: #checked
#             masked = np.ma.MaskedArray(reciprocal_velocities,mask = np.isnan(reciprocal_velocities))
#             reciprocal_avg_velocities = np.ma.average(masked,axis = 0,weights=correlations)
#             # calculate deviations above and below the mean
#             reciprocal_mean_devs = masked - reciprocal_avg_velocities[np.newaxis,...]
#             rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
#             rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
#             N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
#             N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
#             reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
#             reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
#             # take median velocity
#             reciprocal_median_velocities = np.ma.median(masked,axis = 0)
#             # calsulate median absolute deviations
#             reciprocal_median_devs = masked - reciprocal_median_velocities[np.newaxis,...]
#             abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
#             abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
#             reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= 0)
#             reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= 0)
#     # calculate unweighted averages
#     elif not weighted:
#         if time_average and z_average: # checked
#             reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = (0,2))
#             reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = (0,2))
#             # calculate deviations above and below the mean
#             reciprocal_mean_devs = reciprocal_velocities - reciprocal_avg_velocities[np.newaxis,...,np.newaxis]
#             rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
#             rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
#             N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
#             N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
#             reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
#             reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
#             # calsulate median absolute deviations
#             reciprocal_median_devs = reciprocal_velocities - reciprocal_median_velocities[np.newaxis,...,np.newaxis]
#             abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
#             abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
#             reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= (0,2))
#             reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= (0,2))
#         if time_average and not z_average: # checked
#             reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = 2)
#             reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = 2)
#             # calculate deviations above and below the mean
#             reciprocal_mean_devs = reciprocal_velocities - reciprocal_avg_velocities[...,np.newaxis]
#             rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
#             rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
#             N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
#             N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
#             reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
#             reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
#             # calsulate median absolute deviations
#             reciprocal_median_devs = reciprocal_velocities - reciprocal_median_velocities[...,np.newaxis]
#             abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
#             abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
#             reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= 2)
#             reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= 2)
#         if z_average and not time_average: #checked
#             # calculate mean
#             reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = 0)
#             # calculate deviations above and below the mean
#             reciprocal_mean_devs = reciprocal_velocities - reciprocal_avg_velocities[np.newaxis,...]
#             rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
#             rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
#             N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
#             N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
#             reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
#             reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
#             # calculate median
#             reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = 0)
#             # calsulate median absolute deviations
#             reciprocal_median_devs = reciprocal_velocities - reciprocal_median_velocities[np.newaxis,...]
#             abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
#             abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
#             reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= 0)
#             reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= 0)
#     else:
#         print('Weighted bool passed incorrectly. Abort.')
#         print(1/0)
#     # convert from reciprocal space back to normal space
#     avg_velocities = np.divide(1.,reciprocal_avg_velocities)
#     median_velocities = np.divide(1.,reciprocal_median_velocities)
#     # apply correct scaling to the errors
#     pos_stdev = avg_velocities * (reciprocal_pos_stdev/reciprocal_avg_velocities)
#     neg_stdev = avg_velocities * (reciprocal_neg_stdev/reciprocal_avg_velocities)
#     pos_mads = median_velocities * (reciprocal_pos_mads/reciprocal_median_velocities)
#     neg_mads = median_velocities * (reciprocal_neg_mads/reciprocal_median_velocities)
#     return avg_velocities,pos_stdev,neg_stdev,median_velocities,pos_mads,neg_mads