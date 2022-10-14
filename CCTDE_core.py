import numpy as np
from my_funcs import *
import warnings
import numpy.ma as ma
import matplotlib.cm as cm
import multiprocessing as mp
from scipy import signal
from scipy.signal import butter,filtfilt
from scipy.ndimage import gaussian_filter

#This is a test


######################################################################################################################
######################################################################################################################
# Input/Output functions
######################################################################################################################
######################################################################################################################

def load_time_series(SNR,v_angle,velocity,blob_size,IO_parameter,plot_bool=False):
#loads data in dict form. Converts to datacube/np.array. Returns datacube.
    #load_single_pew_dict
    single_pew_dict=load_single_pew_dict(SNR, v_angle, velocity, blob_size,IO_parameter)
    #check if data is empty
    if len(single_pew_dict)>0:
        # initialise parameters for datacube
        x_framesize,y_framesize = np.shape(single_pew_dict[0]['image'])
        time_series_length = len(single_pew_dict)
        #make datacube
        time_series = np.zeros((x_framesize,y_framesize,time_series_length))
        for i in range(time_series_length):
            time_series[:,:,i] = single_pew_dict[i]['image']
    elif len(single_pew_dict)==0:
        time_series = None
    else:
        print('I shat myself. Abort.')
        exit()
    #optional plotting script
    if plot_bool == True and type(time_series) != type(None):
        #check data
        time_step=0.01
        animate =1
        check_single_pew(single_pew_dict,time_step,animate)
    return time_series

def load_single_pew_dict(SNR, v_angle, velocity, blob_size,IO_parameter):
#loads single pew dict. What is there to say?
    #convert parameters to int to avoid confusion
    velocity = int(velocity*1000)
    blob_size = (int(blob_size[0]),int(blob_size[1]))
    if IO_parameter == 0:
        savefolder = '/shared/storage/plasma/turb_exp/ye525/CCTDE/input_data/test/'
        savename = 'vel{0}_sizex{1}y{2}_SNR{3}'.format(velocity, blob_size[0], blob_size[1],SNR)
    elif IO_parameter == 1:
        savefolder = '/shared/storage/plasma/turb_exp/ye525/CCTDE/input_data/Vangle_{0}/'.format(v_angle)
        savename = 'vel{0}_sizex{1}y{2}_SNR{3}'.format(velocity, blob_size[0], blob_size[1],SNR)
    else:
        print('give legit IO parameter')
        print('parameter given:', IO_parameter)
        exit() #exit is invalid IO parameter is given
    return load_obj(savefolder+savename)

def plot_velocity_field(velocity_field):
# plots the velocity field
    warnings.filterwarnings('ignore',category=UserWarning)
    fig,ax = plt.subplots(2,2)
    ax[0,0].imshow(velocity_field[:,:,0].T,origin='lower')
    ax[0,1].imshow(velocity_field[:,:,1].T,origin='lower')
    ax[1,0].imshow(velocity_field[:,:,2].T,origin='lower')
    ax[1,1].imshow(velocity_field[:,:,3].T,origin='lower')

    ax[0,0].set_title('x-velocity field')
    ax[0,1].set_title('x-correlation field')
    ax[1,0].set_title('y-velocity field')
    ax[1,1].set_title('y-correlation field')
    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='grey')
    plt.show()
    return

######################################################################################################################
######################################################################################################################
# checks and balances
######################################################################################################################
######################################################################################################################

def my_butter(mysignal):
    b,a = butter(10,0.3)
    filtered_signal = filtfilt(b,a,mysignal)
    return filtered_signal

def check_single_pew(single_pew_dict,time_step,animate):
#opens up single_pew_dict and allows user to have a peek inside
#prints blob parameters and animates the time_series
    #animation loop
    if animate:
        plt.ion()
        plt.xlabel('X-direction [px]')
        plt.ylabel('Y-direction [px]')
        for i in range(len(single_pew_dict)):
            #extract data from dict
            image = single_pew_dict[i]['image']
            position = single_pew_dict[i]['position']
            in_frame = single_pew_dict[i]['in_frame']
            #print frame number and position
            print('Frame no:{0}. In frame? {2}. Position: {1}.'.format(i,position,in_frame))
            #plot current frame
            plt.title('Frame no: {0}'.format(i))
            plt.contourf(image.T)
            plt.draw()
            plt.pause(time_step)
    #print checked blob parameters
    velocity = single_pew_dict[0]['velocity']
    v_angle = single_pew_dict[0]['v_angle']
    blob_size = single_pew_dict[0]['blob_size']
    SNR = single_pew_dict[0]['SNR']
    #####
    dash = '-'*60
    print(' ')
    print(dash)
    print('Checked blob parameters: ')
    print(dash)
    print('Velocity: ({0}, {1}) [degrees]'.format(velocity,v_angle))
    print('Blob size: {0} [px]'.format(blob_size))
    print('Signal-to-noise: {0}'.format(SNR))
    print(' ')
    return

def input_parameter_checks(ezw,probe_dist,max_tau,no_frames):
#Some simple parameter checks that will break the script.
    if probe_dist>ezw:
        print('probe dist higher than edge-zone width. Abort.')
        exit()
    elif max_tau>no_frames:
        print('Not enough frames you silly. Abort.')
        exit()
    return

def in_edge_zone(i,j,ezw,xn,yn):
# checks if coordinate i,j is in edge zone
# returns True if it is in the edge zone
    if i > xn-ezw-1:
        bool = True
    elif j > yn-ezw-1:
        bool=True
    elif j<ezw:
        bool = True
    elif i < ezw:
        bool = True
    else:
        bool = False
    return bool

def calc_norm_factor(f,g):
#calculates normalisation factor for ccf
    part_f= (np.sum((f)**2))
    part_g= (np.sum((g)**2))
    norm_factor = np.sqrt(part_f * part_g)
    return norm_factor

def crop_time_series(time_series,no_frames,SNR,filtering=False):
# takes in times seried and crops or fills it to length no_frames
# this function should only be used with singleblob data
    tsl=len(time_series[0,0,:])
    half_tsl  = tsl/2
    if tsl<no_frames:
        #pads outsides of time_series with noise frames
        xn,yn = np.shape(time_series[:,:,0])
        if SNR =='inf':
            empty_ts = np.zeros((xn,yn,no_frames))
        else:
            noise_level = 1./SNR
            noise = np.random.normal(0,(noise_level)*20.,(xn,yn,no_frames))
            empty_ts = np.abs(noise)
        # empty_ts = np.zeros((xn,yn,no_frames))
        #account for odd and even tsl
        if tsl%2==0:
            empty_ts[:,:,no_frames/2-half_tsl:no_frames/2+half_tsl]= time_series
        elif tsl%2==1:
            empty_ts[:,:,no_frames/2-half_tsl:no_frames/2+half_tsl+1]= time_series
        time_series = empty_ts
    elif tsl>no_frames:
        #crops time_series to length no_frames
        time_series = time_series[:,:,half_tsl-no_frames/2:half_tsl+no_frames/2]
    else:
        pass
    #noise smoothing
    if filtering:
        gauss_width = 5
        for i in range(no_frames):
            time_series[:,:,i]=gaussian_filter(time_series[:,:,i],gauss_width)
    return time_series

######################################################################################################################
######################################################################################################################
# core cctde functions
######################################################################################################################
######################################################################################################################

def calc_ccf(f,g,max_tau,norm_bool = True,plot_bool=False):
# calculates the cross-correlation function of f and g
# given a certain maximum time delay +-max_tau
    ccf = np.zeros(2*max_tau+1)
    tau_array = np.zeros(2*max_tau+1)
    avg_f,avg_g = (np.mean(f),np.mean(g))
    N_max = float(len(f))
    norm_factor = calc_norm_factor(f,g)
    # plt.ion()
    # fig,ax=plt.subplots(2)
    for i in range(len(ccf)): #loop over all time-delays
        N = len(f)
        tau = i-max_tau
        if tau< 0:
            #calculate cross-correlation value tau<0
            f_part = f[abs(tau):N]-avg_f
            g_part = g[:N-abs(tau)]-avg_g
            fg_multiplied = np.multiply(f_part,g_part)
            fg_N = float(len(fg_multiplied))
            cross_correlation = (N_max/fg_N)*np.sum(np.multiply(f_part,g_part))
        elif tau>= 0:
            #calculate cross-correlation value tau>=0
            f_part = f[:N-abs(tau)]-avg_f
            g_part = g[abs(tau):N]-avg_g
            fg_multiplied = np.multiply(f_part,g_part)
            fg_N = float(len(fg_multiplied))
            cross_correlation = (N_max/fg_N)*np.sum(np.multiply(f_part,g_part))
        #deal with normalisation
        if norm_factor>0.:
            if norm_bool: cross_correlation=cross_correlation/norm_factor
        elif norm_factor==0.:
            cross_correlation = 0.
        else:
            #not sure what to do with norm_factor<0.
            print('norm_factor < 0. This bad? Abort.')
            exit()
        # store cross-correlation value with corresponding tau
        ccf[i] = cross_correlation
        tau_array[i] = tau
    #plot ccf
    if plot_bool==True:
        fig,ax=plt.subplots(3)
        ax[0].plot(tau_array,ccf,'.',ls='-')
        ax[0].set_xlabel('time delay')
        ax[0].set_ylabel('ccf')
        ax[0].set_title('ccf')
        ax[1].plot(f,'.',ls='-')
        ax[1].set_title('f')
        ax[2].plot(g,'.',ls='-')
        ax[2].set_title('g')
        plt.show()
    return ccf,tau_array

def scipy_calc_ccf(f,g,norm_bool = True,plot_bool=False):
# calculates the cross-correlation function of f and g
# given a certain maximum time delay +-max_tau
    #zero center the signals
    f = f - np.mean(f)
    g = g - np.mean(g)
    #calculate unnormalised ccf and lags
    ccf = signal.correlate(g,f,mode='same') #computed time lag from f to g
    N = len(ccf)
    lags = np.linspace(-N//2,N//2-1,N)
    #calculate normalisation factor
    if norm_bool==True:
        norm_factor = calc_norm_factor(f,g)
        ccf = np.divide(ccf,norm_factor)
        for i in range(len(ccf)):
            tau = lags[i]
            lag_norm = N/(N-np.abs(tau))
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
        plt.show()
    return ccf,lags

def calculate_velocity(f,g,probe_dist,correlation_threshold,plot_bool=False):
# takes in two time signatures and calculates the time delays
# at which they have highest correlation.
# With known spatial distance between time signatures and time delay,
# the velocity is calculated and returned.
    #calculate cross-correlation function between f,g
    f = f-np.mean(f)
    g = g-np.mean(g)
    my_ccf,my_tau = scipy_calc_ccf(f,g,norm_bool = True,plot_bool=False)
    if plot_bool:
        fig,ax=plt.subplots(1,2)
        ax[0].plot(f)
        ax[0].plot(g)

        my_acf,my_tau = scipy_calc_ccf(f,f,norm_bool = True,plot_bool=False)
        ax[1].plot(my_tau,my_ccf,ls='--',marker='.',label='ccf')
        ax[1].plot(my_tau,my_acf,label='acf')
        ax[1].set_xlabel('time-delay')
        ax[1].set_ylabel('correlation')
        ax[1].legend()
        plt.show()
        # exit()

    correlation_max = np.max(my_ccf)
    if correlation_max>correlation_threshold:
        #calculate velocity if correlation_max is above a user-defined threshold
        index=np.where(np.max(my_ccf)==my_ccf)[0][0]
        time_delay = my_tau[index]
        #address the possibility of encountering zero time_delay
        if time_delay == 0.:
            #manually set velocity to inf
            velocity = np.nan
        else:
            #calculate velocity
            velocity = probe_dist/time_delay
    else:
        # print('Correlation below threshold: ',correlation_max)
        # print('return NaN')
        velocity = np.nan
    return velocity,correlation_max

def calculate_xy_velocity(time_series,i,j,probe_dist,correlation_threshold,plot_bool=False,filtering=False):
#calculates x- and y-velocity given a reference location i,j
#correlation max values also returned
    #set the probes
    f = time_series[i,j,:]
    xprobe = time_series[i+probe_dist,j,:]
    yprobe = time_series[i,j+probe_dist,:]
    #optional filtering
    if filtering:
        f = my_butter(f)
        xprobe = my_butter(xprobe)
        yprobe = my_butter(yprobe)
    #calculate velocities
    velocities = np.empty(4)
    x_vel,xc_max=calculate_velocity(f,xprobe,probe_dist,correlation_threshold,plot_bool=plot_bool)
    y_vel,yc_max=calculate_velocity(f,yprobe,probe_dist,correlation_threshold,plot_bool=plot_bool)
    velocities = (x_vel,xc_max,y_vel,yc_max)
    return velocities


######################################################################################################################
######################################################################################################################
# example run
######################################################################################################################
######################################################################################################################

datafilename = 'example_input_data'
time_series = load_obj(datafilename)['time_series']
time_series = time_series.astype('float32')
#setting parameters
probe_dist = 20 #controls the spacing between the measurement locations [px]
correlation_threshold = 0.5 #defines the minimum correlation value allowable
#initialise empty velocity field
xn,yn = np.shape(time_series[:,:,0])
velocity_field = np.zeros((xn,yn,4))
#loop over entire fov
for i in range(xn):
    print(i,'/',xn)
    for j in range(yn):
        if in_edge_zone(i,j,probe_dist,xn,yn):
            # exclude velocity measurements in edge zone
            velocity_field[i,j,:] = (np.nan,np.nan,np.nan,np.nan)
        else:
            # measure velocities at ref point i,j
            plot_bool = False
            velocity_field[i,j,:] = calculate_xy_velocity(time_series,i,j,probe_dist,correlation_threshold,plot_bool)

#optional plotting function
vel_field_plot_bool = True
if vel_field_plot_bool:
    plot_velocity_field(velocity_field)
