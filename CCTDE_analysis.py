import numpy as np 
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
print('Note: RuntimeWarnings are currently ignored')

######################################################################################################################
######################################################################################################################
# basic CCTDE analysis functions
######################################################################################################################
######################################################################################################################
def velocity_averaging(velocities,correlations,time_average = True,z_average = False,weighted=True):
    '''
    Averages inferred_velocitied array from vel_scan functions in CCTDE_core.py
    Arguments: (velocities,correlations,time_average = True,z_average = False,weighted=True)
    Returns: avg_velocities,stdevs,median_velocities,mads

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
    weighted: boolean
        weight the averaging according to supplied correlations
        Defaults to True

    Returns:
    --------
    avg_velocities: numpy array
        average velocities.
        shape depends on what axes have been averaged over.
    stdevs: numpy array
        velocity standard deviation
        shape depends on what axes have been averaged over.
    median_velocities: numpy array
        median velocities
        shape depends on what axes have been averaged over.
    mads: numpy array
        median absolute deviations
        shape depends on what axes have been averaged over.

    Notes:
    ------
    :: 
    '''
    # need to take the reciprocal to get proper averaging of the time-lags from CCTDE.
    # straight averaging would give skewed results.
    reciprocal_velocities = np.divide(1.,velocities)
    # calculate weighted averages
    if weighted: 
        if time_average and z_average: #checked
            # mask array where there are nans
            masked = np.ma.MaskedArray(reciprocal_velocities,mask = np.isnan(reciprocal_velocities))
            # take average velocity
            reciprocal_avg_velocities = np.ma.average(masked,axis = (0,2),weights=correlations)
            # calculate deviations above and below the mean
            reciprocal_mean_devs = masked - reciprocal_avg_velocities[np.newaxis,...,np.newaxis]
            rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
            rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
            N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
            N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
            reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
            reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
            # take median velocity
            reciprocal_median_velocities = np.ma.median(masked,axis = (0,2))
            # calsulate median absolute deviations
            reciprocal_median_devs = masked - reciprocal_median_velocities[np.newaxis,...,np.newaxis]
            abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
            abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
            reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= (0,2))
            reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= (0,2))
        if time_average and not z_average: #checked
            masked = np.ma.MaskedArray(reciprocal_velocities,mask = np.isnan(reciprocal_velocities))
            reciprocal_avg_velocities = np.ma.average(masked,axis = 2,weights=correlations)
            # calculate deviations above and below the mean
            reciprocal_mean_devs = masked - reciprocal_avg_velocities[...,np.newaxis]
            rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
            rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
            N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
            N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
            reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
            reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
            # take median velocity
            reciprocal_median_velocities = np.ma.median(masked,axis = 2)
            # calsulate median absolute deviations
            reciprocal_median_devs = masked - reciprocal_median_velocities[...,np.newaxis]
            abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
            abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
            reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= 2)
            reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= 2)
        if z_average and not time_average: #checked
            masked = np.ma.MaskedArray(reciprocal_velocities,mask = np.isnan(reciprocal_velocities))
            reciprocal_avg_velocities = np.ma.average(masked,axis = 0,weights=correlations)
            # calculate deviations above and below the mean
            reciprocal_mean_devs = masked - reciprocal_avg_velocities[np.newaxis,...]
            rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
            rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
            N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
            N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
            reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
            reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
            # take median velocity
            reciprocal_median_velocities = np.ma.median(masked,axis = 0)
            # calsulate median absolute deviations
            reciprocal_median_devs = masked - reciprocal_median_velocities[np.newaxis,...]
            abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
            abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
            reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= 0)
            reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= 0)
    # calculate unweighted averages
    elif not weighted:
        if time_average and z_average: # checked
            reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = (0,2))
            reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = (0,2))
            # calculate deviations above and below the mean
            reciprocal_mean_devs = reciprocal_velocities - reciprocal_avg_velocities[np.newaxis,...,np.newaxis]
            rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
            rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
            N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
            N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
            reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
            reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
            # calsulate median absolute deviations
            reciprocal_median_devs = reciprocal_velocities - reciprocal_median_velocities[np.newaxis,...,np.newaxis]
            abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
            abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
            reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= (0,2))
            reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= (0,2))
        if time_average and not z_average: # checked
            reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = 2)
            reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = 2)
            # calculate deviations above and below the mean
            reciprocal_mean_devs = reciprocal_velocities - reciprocal_avg_velocities[...,np.newaxis]
            rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
            rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
            N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
            N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
            reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
            reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
            # calsulate median absolute deviations
            reciprocal_median_devs = reciprocal_velocities - reciprocal_median_velocities[...,np.newaxis]
            abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
            abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
            reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= 2)
            reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= 2)
        if z_average and not time_average: #checked
            # calculate mean
            reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = 0)
            # calculate deviations above and below the mean
            reciprocal_mean_devs = reciprocal_velocities - reciprocal_avg_velocities[np.newaxis,...]
            rec_pos_vel_dev = np.where(reciprocal_mean_devs>0.,reciprocal_mean_devs,np.nan)
            rec_neg_vel_dev = np.where(reciprocal_mean_devs<0.,reciprocal_mean_devs,np.nan)
            N_pos = np.count_nonzero(~np.isnan(rec_pos_vel_dev))
            N_neg = np.count_nonzero(~np.isnan(rec_neg_vel_dev))
            reciprocal_pos_stdev = np.sqrt(np.nansum(rec_pos_vel_dev**2)/N_pos)
            reciprocal_neg_stdev = np.sqrt(np.nansum(rec_neg_vel_dev**2)/N_neg)
            # calculate median
            reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = 0)
            # calsulate median absolute deviations
            reciprocal_median_devs = reciprocal_velocities - reciprocal_median_velocities[np.newaxis,...]
            abs_pos_median_devs = np.where(reciprocal_median_devs>0.,reciprocal_median_devs,np.nan)
            abs_neg_median_devs = np.abs(np.where(reciprocal_median_devs<0.,reciprocal_median_devs,np.nan))
            reciprocal_pos_mads = np.nanmedian(abs_pos_median_devs,axis= 0)
            reciprocal_neg_mads = np.nanmedian(abs_neg_median_devs,axis= 0)
    else:
        print('Weighted bool passed incorrectly. Abort.')
        print(1/0)
    # convert from reciprocal space back to normal space
    avg_velocities = np.divide(1.,reciprocal_avg_velocities)
    median_velocities = np.divide(1.,reciprocal_median_velocities)
    # apply correct scaling to the errors
    pos_stdev = avg_velocities * (reciprocal_pos_stdev/reciprocal_avg_velocities)
    neg_stdev = avg_velocities * (reciprocal_neg_stdev/reciprocal_avg_velocities)
    pos_mads = median_velocities * (reciprocal_pos_mads/reciprocal_median_velocities)
    neg_mads = median_velocities * (reciprocal_neg_mads/reciprocal_median_velocities)
    return avg_velocities,pos_stdev,neg_stdev,median_velocities,pos_mads,neg_mads

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

def plot_vel_time_one_location(velocities,times, i, j, shotn, N,vlim = 'all',tlim = 'all',plot_average='median'):
    #pick velocity timeseries at one location
    plotting_velocities = velocities[i,j,:]
    #plot velocities against time
    plt.figure(figsize=(10,8))
    plt.plot(times,plotting_velocities,ls='-',marker='.')
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
        plt.hlines(median,np.min(times),np.max(times),linestyles='--',label= 'median: {0:.3f}km/s'.format(median))
        plt.legend()
    elif plot_average == 'mean':
        mean = np.nanmean(plotting_velocities)
        plt.hlines(mean,np.min(times),np.max(times),linestyles='--',label= 'mean: {0:.3f}km/s'.format(mean))
        plt.legend()
    elif plot_average == 'both':
        median = np.nanmedian(plotting_velocities)
        plt.hlines(median,np.min(times),np.max(times),linestyles='--',label= 'median: {0:.3f}km/s'.format(median))
        mean = np.nanmean(plotting_velocities)
        plt.hlines(mean,np.min(times),np.max(times),linestyles='--',label= 'mean: {0:.3f}km/s'.format(mean))
        plt.legend()
    plt.show()
    return

def plot_vel_R_avg_time(velocities,times,correlations,i_range,shotn,N,R,z,vlim = 'all',Rlim = 'all',metric='mean',weighted_avg=True):
    averaged_velocities,pos_stdevs,neg_stdevs,median_velocities,pos_mads,neg_mads = velocity_averaging(velocities,correlations,time_average=True,z_average=False,weighted = weighted_avg)
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
    plt.title('Shotno:{0}, N: {1} \n Time-averaged: [{2:.2f}-{3:.2f}]s'.format(shotn,N,np.min(times),np.max(times)))
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

def plot_vel_R_avg_time_avg_z(velocities,times,correlations,shotn,N,R,vlim = 'all',Rlim = 'all',metric = 'mean', weighted_avg=True):
    averaged_velocities,pos_stdevs,neg_stdevs,median_velocities,pos_mads,neg_mads = velocity_averaging(velocities,correlations,time_average=True,z_average=True,weighted = weighted_avg)
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
    plt.title('Shotno:{0}, N: {1} \n z-averaged, Time-averaged: [{2:.2f}-{3:.2f}]s'.format(shotn,N,np.min(times),np.max(times)))
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