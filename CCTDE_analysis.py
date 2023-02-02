import numpy as np 
import matplotlib.pyplot as plt



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
            # take median velocity
            reciprocal_median_velocities = np.ma.median(masked,axis = (0,2))
            # take standard deviation
            reciprocal_stdevs = np.ma.std(masked,axis = (0,2))
            # calsulate median absolute deviation
            abs_median_devs = np.abs(masked - reciprocal_median_velocities[np.newaxis,...,np.newaxis])
            reciprocal_mads = np.ma.median(abs_median_devs,axis = (0,2))
        if time_average and not z_average: #checked
            masked = np.ma.MaskedArray(reciprocal_velocities,mask = np.isnan(reciprocal_velocities))
            reciprocal_avg_velocities = np.ma.average(masked,axis = 2,weights=correlations)
            reciprocal_stdevs = np.ma.std(masked,axis = 2)
            reciprocal_median_velocities = np.ma.median(masked,axis = 2)
            abs_median_devs = np.abs(masked - reciprocal_median_velocities[...,np.newaxis])
            reciprocal_mads = np.ma.median(abs_median_devs,axis = 2)
        if z_average and not time_average: #checked
            masked = np.ma.MaskedArray(reciprocal_velocities,mask = np.isnan(reciprocal_velocities))
            reciprocal_avg_velocities = np.ma.average(masked,axis = 0,weights=correlations)
            reciprocal_median_velocities = np.ma.median(masked,axis = 0)
            reciprocal_stdevs = np.ma.std(masked,axis = 0)
            abs_median_devs = np.abs(masked - reciprocal_median_velocities[np.newaxis,...])
            reciprocal_mads = np.ma.median(abs_median_devs,axis = 0)
    # calculate unweighted averages
    elif not weighted:
        if time_average and z_average: # checked
            reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = (0,2))
            reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = (0,2))
            reciprocal_stdevs = np.nanstd(reciprocal_velocities,axis = (0,2))
            abs_median_devs = np.abs(reciprocal_velocities - reciprocal_median_velocities[np.newaxis,...,np.newaxis])
            reciprocal_mads = np.nanmedian(abs_median_devs,axis = (0,2))
        if time_average and not z_average: # checked
            reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = 2)
            reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = 2)
            reciprocal_stdevs = np.nanstd(reciprocal_velocities,axis = 2)
            abs_median_devs = np.abs(reciprocal_velocities - reciprocal_median_velocities[...,np.newaxis])
            reciprocal_mads = np.nanmedian(abs_median_devs,axis = 2)
        if z_average and not time_average: #checked
            reciprocal_avg_velocities = np.nanmean(reciprocal_velocities,axis = 0)
            reciprocal_median_velocities = np.nanmedian(reciprocal_velocities,axis = 0)
            reciprocal_stdevs = np.nanstd(reciprocal_velocities,axis = 0)
            abs_median_devs = np.abs(reciprocal_velocities - reciprocal_median_velocities[np.newaxis,...])
            reciprocal_mads = np.nanmedian(abs_median_devs,axis = 0)
    else:
        print('Weighted bool passed incorrectly. Abort.')
        print(1/0)
    # convert from reciprocal space back to normal space
    avg_velocities = np.divide(1.,reciprocal_avg_velocities)
    median_velocities = np.divide(1.,reciprocal_median_velocities)
    # apply correct scaling to the errors
    stdevs = avg_velocities * (reciprocal_stdevs/reciprocal_avg_velocities)
    mads = median_velocities * (reciprocal_mads/reciprocal_median_velocities)
    return avg_velocities,stdevs,median_velocities,mads


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
    plt.title('Shotno:{0}, N: {3}, Velocity from channel {1} to {2}'.format(shotn,(i,j),(i+1,j),N))
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
    averaged_velocities,stdevs,median_velocities,mads = velocity_averaging(velocities,correlations,time_average=True,z_average=False,weighted = weighted_avg)
    plt.figure(figsize=(10,8))
    if metric == 'mean':
        for i in i_range:
            plt.plot(R[i,:],averaged_velocities[i,:],marker='.',label = 'z:{:.2f}m'.format(z[i,0]))
            plt.fill_between(R[i,:],averaged_velocities[i,:]-stdevs[i,:],averaged_velocities[i,:]+stdevs[i,:],alpha=0.2)
            plt.ylabel('Mean inferred velocity [km/s]')
    elif metric == 'median':
        for i in i_range:
            plt.plot(R[i,:],median_velocities[i,:],marker='.',label = 'z:{:.2f}m'.format(z[i,0]))
            plt.fill_between(R[i,:],median_velocities[i,:]-mads[i,:],median_velocities[i,:]+mads[i,:],alpha=0.2)
            plt.ylabel('Median inferred velocity [km/s]')
    elif metric == 'both':
        for i in i_range:
            plt.plot(R[i,:],averaged_velocities[i,:],marker='.',label = 'mean, z:{:.2f}m'.format(z[i,0]))
            plt.fill_between(R[i,:],averaged_velocities[i,:]-stdevs[i,:],averaged_velocities[i,:]+stdevs[i,:],alpha=0.2)
            plt.plot(R[i,:],median_velocities[i,:],marker='.',label = 'median, z:{:.2f}m'.format(z[i,0]))
            plt.fill_between(R[i,:],median_velocities[i,:]-mads[i,:],median_velocities[i,:]+mads[i,:],alpha=0.2)
            plt.ylabel('Inferred velocity [km/s]')
    plt.xlabel('Major radius [m]')
    plt.title('Shotno:{0}, N: {1}, Time-averaged: [{2:.2f}-{3:.2f}]s'.format(shotn,N,np.min(times),np.max(times)))
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
    averaged_velocities,stdevs,median_velocities,mads = velocity_averaging(velocities,correlations,time_average=True,z_average=True,weighted = weighted_avg)
    plt.figure(figsize=(10,8))
    if metric == 'mean':
        plt.plot(R[0,:],averaged_velocities,marker='.',label = 'mean',color='b')
        plt.fill_between(R[0,:],averaged_velocities-stdevs,averaged_velocities+stdevs,alpha=0.2,color='b')
    elif metric == 'median':
        plt.plot(R[0,:],median_velocities,marker='.',label = 'median',color = 'orange')
        plt.fill_between(R[0,:],median_velocities-mads,median_velocities+mads,alpha=0.2,color = 'orange')
    elif metric == 'both':
        plt.plot(R[0,:],averaged_velocities,marker='.',label = 'mean',color = 'b')
        plt.fill_between(R[0,:],averaged_velocities-stdevs,averaged_velocities+stdevs,alpha=0.2,color = 'b')
        plt.plot(R[0,:],median_velocities,marker='.',label = 'median',color = 'orange')
        plt.fill_between(R[0,:],median_velocities-mads,median_velocities+mads,alpha=0.2,color = 'orange')
    plt.xlabel('Major radius [m]')
    plt.ylabel('Inferred velocity [km/s]')
    plt.title('Shotno:{0}, N: {1}, z-averaged, Time-averaged: [{2:.2f}-{3:.2f}]s'.format(shotn,N,np.min(times),np.max(times)))
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