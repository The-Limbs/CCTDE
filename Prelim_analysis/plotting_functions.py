import BES
import pyuda
try:
    from cpyuda import ServerException as SE
except ModuleNotFoundError:
    print('cpyuda.ServerException couldnt be imported')
import numpy as np
import matplotlib.pyplot as plt
import os

###########################################################################################
###########################################################################################
# helper/minor functions
###########################################################################################
###########################################################################################

def find_nearest(arr, val):
    return np.abs(arr - val).argmin()

def getCoreNe(shotn, prnt=True):
    client = pyuda.Client()
    if prnt:
        print('loading core line-averaged density')
    try:
        ne = client.get('/xdc/ai/raw/density', shotn)
        if prnt:
            print('loaded /xdc/ai/raw/density')
    except SE:
        try:
            ne = client.get('/ANE/DENSITY', shotn)
            if prnt:
                print('loaded /ANE/DENSITY')
        except SE:
            try:
                ne = client.get('ANE_DENSITY', shotn)
                if prnt:
                    print('loaded ANE_DENSITY')
            except SE:
                if prnt:
                    print('core line-averaged density could not be loaded')
                ne = None
    return ne

def getIp(shotn, prnt=True):
    client = pyuda.Client()
    if prnt:
        print('Loading plasma current')
    try:
        ip = client.get('ip', shotn)
        if prnt:
            print('loaded ip')
    except SE:
        try:
            ip = client.get('/AMC/PLASMA_CURRENT', shotn)
            if prnt:
                print('loaded /AMC/PLASMA_CURRENT')
        except SE:
            try:
                ip = client.get('/XDC/AI/CPU1/PLASMA_CURRENT',
                                shotn)
                if prnt:
                    print('loaded /XDC/AI/CPU1/PLASMA_CURRENT')
            except SE:
                try:
                    ip = client.get('/XDC/AI/CPU2/PLASMA_CURRENT',
                                    shotn)
                    if prnt:
                        print('loaded /XDC/AI/CPU2/PLASMA_CURRENT')
                except SE:
                    if prnt:
                        print('plasma current could not be loaded')
                    ip = None
    return ip

def getCoreTe(shotn, prnt=True):
    client = pyuda.Client()
    if prnt:
        print('loading core temperature')
    try:
        te = client.get('/AYC/T_E_CORE', shotn)
        if prnt:
            print('loaded /AYC/T_E_CORE')
    except SE:
        try:
            te = client.get('AYC_TE_CORE', shotn)
            if prnt:
                print('loaded AYC_TE_CORE')
        except SE:
            if prnt:
                print('core temperature could not be loaded')
            te = None
    return te

def getDalpha(shotn, prnt=True):
    client = pyuda.Client()
    if prnt:
        print('loading radial D-alpha')
    try:
        dar = client.get('/XIM/DA/HM10/R', shotn)
        if prnt:
            print('loaded /XIM/DA/HM10/R')
    except SE:
        try:
            dar = client.get('XIM_DA/HM10/R', shotn)
            if prnt:
                print('loaded XIM_DA/HM10/R')
        except SE:
            if prnt:
                print('radial D-alpha could not be loaded')
            dar = None
    if prnt:
        print('loading tangential D-alpha')
    try:
        dat = client.get('/XIM/DA/HM10/T', shotn)
        if prnt:
            print('loaded /XIM/DA/HM10/T')
    except SE:
        try:
            dat = client.get('XIM_DA/HM10/T', shotn)
            if prnt:
                print('loaded XIM_DA/HM10/T')
        except SE:
            if prnt:
                print('tangential D-alpha could not be loaded')
            dat = None
    return dar, dat

def getBeamPower(shotn, prnt=True):
    client = pyuda.Client()
    if prnt:
        print('loading SS beam power')
    try:
        beamSS = client.get('/XNB/SS/BEAMPOWER', shotn)
        if prnt:
            print('loaded /XNB/SS/BEAMPOWER')
    except SE:
        try:
            beamSS = client.get('ANB_SS_SUM_POWER', shotn)
            if prnt:
                print('loaded ANB_SS_SUM_POWER')
        except SE:
            if prnt:
                print('SS beam power could not be loaded')
            beamSS = None
    if prnt:
        print('loading SW beam power')
    try:
        beamSW = client.get('/XNB/SW/BEAMPOWER', shotn)
        if prnt:
            print('loaded /XNB/SW/BEAMPOWER')
    except SE:
        try:
            beamSW = client.get('ANB_SW_SUM_POWER', shotn)
            if prnt:
                print('loaded ANB_SW_SUM_POWER')
        except SE:
            if prnt:
                print('SW beam power could not be loaded')
            beamSW = None
    return beamSS, beamSW

def getBeamDuration(shotn, beam='SS', limit=0.2, prnt=True):
    if not beam in ['SS', 'SW']:
        print('beam must be either SS or SW')
        print('defaulting to SS')
        beam = 'SS'
    if beam == 'SS':
        power = getBeamPower(shotn, prnt=prnt)[0]
    else:
        power = getBeamPower(shotn, prnt=prnt)[1]
    textra = getBeamAdjust(shotn, beam=beam, prnt=prnt) # TODO: the kwargs for fn
    try:
        t1 = power.time.data[np.where(power.data > limit)[0][0]]
        t2 = power.time.data[np.where(power.data > limit)[0][-1]]
    except AttributeError:
        print('AttributeError, presumably unable to get beam data')
        t1 = np.nan
        t2 = np.nan
    except IndexError:
        print('IndexError, presumably unable to get beam data')
        t1 = np.nan
        t2 = np.nan
    return t1 + textra, t2 + textra

def getBeamAdjust(shotn, beam='SS', limit=0.5, refLength=0.015, refTime=0.004995, prnt=True):
    # check stuff for beam
    if not beam in ['SS', 'SW']:
        print('beam must be either SS or SW')
        print('defaulting to SS')
        beam = 'SS'
    # get beam time
    try:
        if beam == 'SS':
            time = getBeamPower(shotn, prnt=prnt)[0].time.data
        else:
            time = getBeamPower(shotn, prnt=prnt)[1].time.data
    except AttributeError:
        print('No beams for shot {}, shift set to zero'.format(shotn))
        return 0.
    # get TSSIG signal
    client = pyuda.Client()
    try:
        tssig = client.get('/XNB/{}/TSSIG'.format(beam), shotn).data
    except SE:
        print('No TSSIG for {} beam, shift set to zero')
        return 0.
    try:
        ### find dt, but double so as to avoid minor rounding error in time
        dt = np.diff(time).mean() * 2.
        ### find when the beam is off
        beamOff = time[tssig < (tssig.max() * limit)]
        ### difference between beam off times
        timeDiff = np.diff(beamOff)
        ### length of a pulse of the beam
        pulseLength = timeDiff[timeDiff > dt] # all done
        ### time of the pulse of the beam
        pulseTime = beamOff[:-1][timeDiff > dt] # all done
        ### get index of pulseLength closest to reference
        pindex = find_nearest(pulseLength, refLength)
        ### calculate the shift
        shift = refTime - pulseTime[pindex] # all done
    except ValueError:
        print('assume this hasnt worked, setting to zero')
        shift = 0.
    return shift


###########################################################################################
###########################################################################################
# main plotting functions
###########################################################################################
###########################################################################################

def subplotsBES(shotn, data=None, time=None, R=None, z=None,
                lim_BES=[-0.1, 2.1], lim_Ip=[-150, 1200],
                lim_Ne=[-0.2, 4.2], lim_Te=[-100, 1900],
                lim_Dalpha=[-0.06, 0.4], lim_NBI=[-0.1, 2.4],
                lim_time=[-0.2, 1.4], i=4, j=0, savePath = 'undef', prnt=True):
    '''
    Plots a number of basic traces (such as ne,Ip,BES raw) for a given shotno.
    Arguments: (shotn, data=None, time=None, R=None, z=None,
                lim_BES=[-0.1, 2.1], lim_Ip=[-150, 1200],
                lim_Ne=[-0.2, 4.2], lim_Te=[-100, 1900],
                lim_Dalpha=[-0.06, 0.4], lim_NBI=[-0.1, 2.4],
                lim_time=[-0.2, 1.4], i=4, j=0, savePath = 'undef', prnt=True)
    Returns: nothing

    Parameters:
    -----------
    shotn: integer
        MAST (U) shotnumber
    
    Keyword arguments:
    ------------------
    data: 1D numpy array
        BES data to be plotted
        if None is passed, data will be imported
    time: 1D numpy array
        time data to be plotted
        if None is passed, data will be imported
    R,z : floats
        Spatial coordinates of the BES channel
        if None is passed, data will be imported
    i,j : integers
        Channel numbers of the BES channels
    lim_{var} : list of two floats
        Plotting limits of respective variables.
        lim_time = 'beamplot' can be passed to only plot the times when the beam is on
    savePath: string
        path to which the plots shall be saved. WATCH OUT. Directory will be made automatically  
        savePath = 'undef' can be passed if saving is undesired.
    prnt: Boolean
        True if log of data import is to be printed in stdout
        False if you like a clean terminal
    
    Notes:
    ------
    :: This function was taken from Steven Thomas' bes git repository. It was then cleaned and annotated. 
    :: Function should be highly portable now. Only BES.py is required in same directory.
    '''
    #import the BES data if none given.
    try:
        if data is None:
            # data, time, R, z = getBESsingle(shotn, I=i, J=j, offset=True)
            data, time, R, z = BES.getSingle(shotn, I=i, J=j, offset=True)
            # data, time, R, z = getBES(shotn, splitRz=True)
            # data = subtractOffset(shotn, data, time)
        if R is None or z is None:
            R, z = BES.getRz(shotn)
        BESbool = True
    except IndexError:
        BESbool = False
    #import Ne,Ip,Te,Dalpha,beam data
    ne = getCoreNe(shotn, prnt=prnt)
    Ip = getIp(shotn, prnt=prnt)
    Te = getCoreTe(shotn, prnt=prnt)
    dar, dat = getDalpha(shotn, prnt=prnt)
    bSS, bSW = getBeamPower(shotn, prnt=prnt)
    t1, t2 = getBeamDuration(shotn, beam='SS', prnt=prnt)
    tSS = getBeamAdjust(shotn, beam='SS', prnt=prnt)
    tSW = getBeamAdjust(shotn, beam='SW', prnt=prnt)
    ts = [t1, t2]
    #initialise figure
    fig, ax = plt.subplots(6, 1, sharex=True,
                           figsize=(6,10), dpi=120)

    #################### start of plotting ####################
    #plot BES data if there is any
    if BESbool:
        ax[0].plot(time, data, c='C0')
        fig.suptitle('Shot # {}, Rz=[{:.3f},{:.3f}]m'
                     .format(shotn, R, z))
    else:
        fig.suptitle('Shot # {}'.format(shotn))
    ax[0].set_ylabel('BES raw (arb)')
    ax[0].grid()
    ax[0].set_ylim(lim_BES)
    #plot Ip if possible
    try:
        ax[1].plot(Ip.time.data, Ip.data, c='C0')
    except AttributeError:
        pass
    ax[1].set_ylabel('$I_p$ (kA)')
    ax[1].grid()
    ax[1].set_ylim(lim_Ip)
    #plot Ne if possible
    try:
        if ne.data.max() > 1e19:
            ax[2].plot(ne.time.data, ne.data / 1e20, c='C0')
        else:
            ax[2].plot(ne.time.data, ne.data, c='C0')
    except AttributeError:
        pass
    ax[2].set_ylabel('$\\bar{{n}}_e$ ($\\times10^{{20}}$m$^{{-2}}$)')
    ax[2].grid()
    ax[2].set_ylim(lim_Ne)
    #plot Te if possible
    try:
        ax[3].plot(Te.time.data, Te.data, c='C0')
    except AttributeError:
        pass
    ax[3].set_ylabel('Te (eV)')
    ax[3].grid()
    ax[3].set_ylim(lim_Te)
    #plot Dalpha if possible
    try:
        ax[4].plot(dar.time.data, dar.data, c='C0')
    except AttributeError:
        pass
    try:
        ax[4].plot(dat.time.data, dat.data, c='C1')
    except AttributeError:
        pass
    ax[4].set_ylabel('Dalpha (V)')
    ax[4].set_ylim(lim_Dalpha)
    ax[4].grid()
    #plot NBI power if possible
    try:
        ax[5].plot(bSS.time.data+tSS, bSS.data, c='C0')
    except AttributeError:
        pass
    try:
        ax[5].plot(bSW.time.data+tSW, bSW.data, c='C1')
    except AttributeError:
        pass
    ax[5].grid()
    ax[5].set_ylabel('NBI Power (MW)')
    ax[5].set_ylim(lim_NBI)
    #################### plot formatting ####################
    # set time limits
    if lim_time is 'beamplot':
        try:
            ax[-1].set_xlim([t1-0.02,t2+0.02])
        except ValueError:
            ax[-1].set_xlim([-0.2, 1.4])
    else:
        ax[-1].set_xlim(lim_time)
    ax[-1].set_xlabel('time (s)')
    # plot vlines to show beam duration
    if not np.isnan(ts[0]):
        ax[0].vlines(ts, lim_BES[0], lim_BES[1], colors=['C2','C3'])
        ax[1].vlines(ts, lim_Ip[0], lim_Ip[1], colors=['C2','C3'])
        ax[2].vlines(ts, lim_Ne[0], lim_Ne[1], colors=['C2','C3'])
        ax[3].vlines(ts, lim_Te[0], lim_Te[1], colors=['C2','C3'])
        ax[4].vlines(ts, lim_Dalpha[0], lim_Dalpha[1], colors=['C2','C3'])
        ax[5].vlines(ts, lim_NBI[0], lim_NBI[1], colors=['C2','C3'])
    if savePath != 'undef':
        # make save directory if it doesn't exist
        if not os.path.exists(savePath + str(shotn)):
            os.makedirs(savePath + str(shotn))
        if lim_time is 'beamplot':
            plt.savefig(savePath + '{}/{}_subplots_{}_{}.png'.format(shotn,shotn,i,j))
        else:
            plt.savefig(savePath + '{}/{}_subplots_beam_{}_{}.png'.format(shotn,shotn,i,j))
    else:
        print('savePath undefined. Did not save figure.')
        pass   
    return

print('done!')