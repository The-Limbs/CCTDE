# import BESMap
import flap
from flap_apdcam import flap_apdcam
from cpyuda import ServerException as SE
import numpy as np
import pyuda
from scipy.signal import butter, filtfilt
import warnings
import sys
#sys.path.insert(0, '/home/sthoma/bes/')
import BESMap

### SThomas change 3rdJan2023
### make global variables here as multiple functions
### use them and it's easier to change them here should
### I need to
sep = 0.01812
chip0 = 2.3e-3
chip1 = 2.6e-3
chip2 = 2.15e-3

def processBES(shotn, data=None, time=None, I=None, J=None,
               trange=[-1.,-1], limitNBI=0.2, NNBI=5000,
               dnNBI=10, cutoffNBI=8.3e3, cutoffHP=8.3e3,
               cutoffLP=5e5, meanSubtract=True, order=3):
    """
    Process the BES signal

    Parameters
    ----------
    shotn : the shot number, int
    data : the BES data, numpy array, with time aligned
           along the final axis
    time : the time of the BES data, 1D numpy array
    limitNBI : the limit used to find the start of the NBI
    NNBI : number of points used for subtracting offset
           from before the NBI starts
    dnNBI : number of points before NBI start time for offset
    cutoffNBI=8.3e3 : lowpass freqCO used for NBI fluctuations
    cutoffHP=8.3e3 : highpass freqCO for removing MHD
    cutoffLP=5e5 : lowpass freqCO for removing highf noise
    meanSubtract=True : boolean, remove the NBI contribution
    order=3 : order of the butter_filter used

    Returns
    -------
    All returns are the same shape as data
    data : the raw data
    time : the time for all data arrays
    dataSubtracted :
    spikeLimit :
    dataSpiked :
    spikeFractions :
    dataFluctHPLP : I - <I>, fluctuations, bandpassed
    dataFluctRel : (I - <I>) / <I> - fluctuations,
                 bandpassed, divided by NBIlowpass
    dataFluct : I - <I>, not filtered
    dataLP0 : lowpass of raw data at high f
    dataNBILP : <I> lowpass of raw data at NBI f
    """
    if (data is None) or (time is None):
        if (I is not None) and (J is not None):
            data, time, _, _ = getSingle(shotn, I=I, J=J)
        else:
            data, time, _, _ = get(shotn)
    dataSubtracted = subtractOffset(shotn, data, time,
                                    limit=limitNBI, N=NNBI,
                                    dn=dnNBI, axis=-1)
    spikeLimit = calcSpikeLimit(dataSubtracted, axis=-1)
    if data.ndim > 1:
        dataSpiked = np.zeros_like(dataSubtracted)
        spikeFractions = np.zeros(dataSubtracted.shape[:-1])
        for i in range(0, dataSpiked.shape[0]):
            for j in range(0, dataSpiked.shape[1]):
                dataSpiked[i,j,:], spikeFractions[i,j] = \
                removeSpikes(dataSubtracted[i,j,:], spikeLimit)
    else:
        dataSpiked, spikeFractions = removeSpikes(dataSubtracted, spikeLimit)
    fs = 1. / np.mean(np.diff(time))
    dataFluctHPLP, dataFluctRel, dataFluct, dataLP0, dataNBILP = \
                   filterBES(dataSpiked, fs, cutoffNBI=cutoffNBI,
                   cutoffHP=cutoffHP, cutoffLP=cutoffLP,
                   meanSubtract=meanSubtract, order=order, axis=-1)
    if (trange[0] == -1.) and (trange[1] == -1.):
        tind = np.ones(len(time)).astype(bool)
    else:
        tind = (time >= trange[0]) * (time <= trange[1])
    return data[...,tind], time[tind], dataSubtracted[...,tind], \
           spikeLimit, dataSpiked[...,tind], spikeFractions, \
           dataFluctHPLP[...,tind], dataFluctRel[...,tind], \
           dataFluct[...,tind], dataLP0[...,tind], \
           dataNBILP[...,tind]

def getSingle(shotn, trange=[-1.,-1.], I=4, J=4, trigger=0.1, offset=False, N=5000):
    if shotn < 45177:
        client = pyuda.Client()
        time = client.get('/xbt/channel01', shotn).time.data # [s]
        sig = client.get('/xbt/channel' + str(32-(I*8)-J)/zfill(2))
    else:
        flap_apdcam.register()
        fname, _ = getInfo(shotn)
        datapath = '/home/sthoma/RawData/{}/'.format(fname)
        if shotn >= 45381:
            frequency = 4e6 # [4MHz]
        else:
            frequency = 2e6 # [2MHz]

        if (shotn <= 45500) or (shotn >= 46451):
            name = 'ADC' + str(getPos()[I,J])
        else:
            name = 'Pixel_{}-{}'.format(J,I)
        try:
            time = flap.get_data('APDCAM', name=name, options=
                             {'Datapath':datapath,'Scaling':'Volts'}
                              ).coordinate('Time')[0]
        except OSError:
            try:
                datapath = '/common/projects/diagnostics/MAST/FIDA/'\
                           'BES_data/{}/'.format(fname)
                time = flap.get_data('APDCAM', name=name, options=
                                 {'Datapath':datapath,'Scaling':'Volts'}
                                  ).coordinate('Time')[0]
            except OSError:
                datapath = '/home/sthoma/freia/RawData/{}/'.format(fname)
                time = flap.get_data('APDCAM', name=name, options=
                                 {'Datapath':datapath,'Scaling':'Volts'}
                                  ).coordinate('Time')[0]
        time -= trigger
        sig = flap.get_data('APDCAM', name=name,
              options={'Datapath':datapath, 'Scaling':'Volt'}).data
    if (trange[0] == -1.) and (trange[1] == -1.):
        tinds = np.ones(len(time)).astype(bool)
    else:
        tinds = (time >= trange[0]) * (time <= trange[1])
    R, z = getRz(shotn)
    R = R[I,J]
    z = z[I,J]
    if offset:
        sig -= np.mean(sig[:N])
    return sig[tinds], time[tinds], R, z

def get(shotn, trange=[-1.,-1.], splitRz=True, pcolor=False, trigger=0.1, offset=False, N=5000):
    if shotn < 45177:
        client = pyuda.Client()
        time = client.get('/xbt/channel01', shotn).time.data # [s]
        data = np.zeros((4,8,len(time))) # empty array for data
        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[1]):
                name = '/xbt/channel' + str(32-(i*8)-j).zfill(2)
                data[i,j] = client.get(name, shotn).data
    else:
        flap_apdcam.register()
        fname, _ = getInfo(shotn)
        datapath = '/common/projects/diagnostics/MAST/FIDA/'\
                   'BES_data/{}/'.format(fname)
        if shotn >= 45381:
            frequency = 4e6 # [4MHz]
        else:
            frequency = 2e6 # [2MHz]
        if (shotn <= 45500) or (shotn >= 46451):
            name = 'ADC33'
            prefix = 'ADC'
        else:
            name = 'Pixel_1-1'
            prefix = 'Pixel_'
        try:
            time = flap.get_data('APDCAM', name=name, options=
                             {'Datapath':datapath,'Scaling':'Volts'}
                              ).coordinate('Time')[0]
        except OSError:
            try:
                datapath = '/home/sthoma/RawData/{}/'.format(fname)
                time = flap.get_data('APDCAM', name=name, options=
                                 {'Datapath':datapath,'Scaling':'Volts'}
                                  ).coordinate('Time')[0]
            except OSError:
                datapath = '/home/sthoma/freia/RawData/{}/'.format(fname)
                time = flap.get_data('APDCAM', name=name, options=
                                 {'Datapath':datapath,'Scaling':'Volts'}
                                  ).coordinate('Time')[0]
        time -= trigger
        T = len(time)
        # end of changes
        data = np.zeros((8,8,T)).astype(np.float64)
        # Rz = np.zeros((8,8,2))
        if (shotn <= 45500) or (shotn >= 46451):
            APDmap = getPos()
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    name = prefix + str(APDmap[i,j])
                    data[i,j] = flap.get_data('APDCAM', name=name,
                                options={'Datapath':datapath,
                                'Scaling':'Volt'}).data
        else:
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    name = prefix + '{}-{}'.format(j,i)
                    data[i,j] = flap.get_data('APDCAM', name=name,
                                options={'Datapath':datapath,
                                'Scaling':'Volt'}).data
    if offset:
        data -= np.mean(data[...,:N], axis=-1, keepdims=True)
    if (trange[0] == -1.) and (trange[1] == -1.):
        tinds = np.ones(len(time)).astype(bool)
    else:
        tinds = (time >= trange[0]) * (time <= trange[1])
    if pcolor:
        R, z = getRzpcolor(shotn, splitRz=True)
    else:
        # Rz = getRz(shotn, splitRz=False)
        R, z = getRz(shotn) # todo needs chaning
    if splitRz:
        # return data[:,:,tinds], time[tinds], Rz[:,:,0], Rz[:,:,1]
        return data[:,:,tinds], time[tinds], R, z
    else:
        # return data[:,:,tinds], time[tinds], Rz
        return data[:,:,tinds], time[tinds], np.append(R[:,:,np.newaxis], z[:,:,np.newaxis], axis=-1)

def subtractOffset(shotn, data, time, limit=0.2,
                   N=5000, dn=10, axis=-1, tbackup=-0.01):
    t1 = getBeamDuration(shotn, beam='SS', limit=limit)[0] # beam on
    if not np.isnan(t1): # if beam is on
        I = find_nearest(time, t1) # find where t=t1
    else: # if the beam is not on
        I = find_nearest(time, tbackup) # find where t=t1
    ### offset of data before the beam
    offset = np.mean(data[...,I-N-dn:I-dn], axis=axis)
    data = data - np.expand_dims(offset, axis) # take off offset
    return data

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

def calcSpikeLimit(data, axis=-1):
    mean_dc = np.mean(data, axis=axis)
    mean_dc_max = mean_dc.max()
    ### limits at which no points are filtered
    ### in the LED signal vs its DC level
    dc = [0, 0.65, 0.83, 1, 1.53]
    limit = [0.014, 0.018, 0.027, 0.025, 0.039]
    ### using linear regression for
    ### approximation of the limit value
    limit_appr = np.polyfit(dc, limit, 1, full=True)
    limit = limit_appr[0][0] * mean_dc_max + limit_appr[0][1]
    return limit

def removeSpikes(sig, limit):#
    ### TODO ### needs adjusting to be like Alsu's
    ### zscore=0.7, n=5, nmean=None, absDiff=False
    sig2 = sig.copy()
    differential = np.diff(sig)
    for i, (x, y, z) in enumerate(zip(differential,
                        differential[1:], differential[2:])):
        if x > limit and y < -limit:
            sig2[i+1] = sig2[i]
        if x > limit and z < -limit:
            sig2[i+1] = sig2[i]
            try:
                sig2[i+2] = sig2[i+1]
            except(IndexError, TypeError, ValueError):
                sig2[i+2] = sig2[i]
    sp1 = sig - sig2
    msk = sp1 > 0
    spikeFraction = np.sum(msk) / len(msk)
    return sig2, spikeFraction

def filterBES(data, fs, cutoffNBI=8.3e3, cutoffHP=8.3e3,
              cutoffLP=5e5, meanSubtract=True, order=3, axis=-1):
    """
    Filtering the BES signals

    Parameters
    ----------
    data : the BES data, spikes already removed
    fs : sampling frequency
    cutoffNBI=8.3e3 : lowpass freqCO used for NBI fluctuations
    cutoffHP=8.3e3 : highpass freqCO for removing MHD
    cutoffLP=5e5 : lowpass freqCO for removing highf noise
    meanSubtract=True : boolean, remove the NBI contribution
    order=3 : order of the butter_filter used
    axis=-1 : axis along which the time should be aligned

    Returns
    -------
    All returns are the same shape as data
    dataFluctHPLP : I - <I>, fluctuations, bandpassed
    dataFluctRel : (I - <I>) / <I> - fluctuations,
                 bandpassed, divided by NBIlowpass
    dataFluct : I - <I>, not filtered
    dataLP0 : lowpass of raw data at high f
    dataNBILP : <I> lowpass of raw data at NBI f
    """
    data2 = data.copy()
    if meanSubtract:
        # fluctuation1 - data minus the mean
        data2 = data - np.mean(data, axis=axis, keepdims=True)
        ### lowpass below the frequency of
        ### the NBI of fluctuation1 + mean
        dataNBILP = butter_filter(data2, cutoffNBI, fs,
                    'lowpass', order=order, analog=False,
                    axis=axis) + np.mean(data, axis=axis,
                    keepdims=True)
        # fluctuation2 = rawdata - lowpassNBI
        dataFluct = data - dataNBILP
        # lowpass of rawdata under highest frequency
        dataLP0 = butter_filter(data, cutoffLP, fs,
                                'lowpass', order=order,
                                analog=False, axis=axis)
        # lowpass of fluctuation2 under highest frequency
        dataFluctLP = butter_filter(dataFluct, cutoffLP, fs,
                                    'lowpass', order=order,
                                    analog=False, axis=axis)
        # highpass of the one above
        dataFluctHPLP = butter_filter(dataFluctLP, cutoffHP, fs,
                                      'highpass', order=order,
                                      analog=False, axis=axis)
        # bandpassed data / NBIfiltered
        dataFluctRel = dataFluctHPLP / dataNBILP
    else:
        # lowpass below the NBI frequency of rawdata
        dataNBILP = butter_filter(data2, cutoffNBI, fs,
                                  'lowpass', order=order,
                                  analog=False, axis=axis)
        # equivalent of fluctuation2, rawdata - lowpassNBI
        dataFluct = data - dataNBILP
        # lowpass of rawdata under highest frequency
        dataLP0 = butter_filter(data2, cutoffLP, fs,
                                'lowpass', order=order,
                                analog=False, axis=axis)
        # highpass of the one above
        # this one is different because it uses lowpass of rawdata
        # where as above uses lowpass of rawdata - NBI
        dataFluctHPLP = butter_filter(dataLP0, cutoffHP, fs,
                                      'highpass', order=order,
                                      analog=False, axis=axis)
        # bandpassed data / NBIfiltered
        dataFluctRel = dataFluctHPLP / dataNBILP
    return dataFluctHPLP, dataFluctRel, \
           dataFluct, dataLP0, dataNBILP

def getInfo(shotn):
    mapping = BESMap.BESMap()
    I = np.where(mapping[0] == str(shotn))[0][0]
    fname = str(mapping[1,I])
    radius = float(mapping[2,I])
    return fname, radius

def getPos():
    mapping = np.array([[33, 9, 24, 22, 60, 58, 39, 16],
                        [10, 34, 23, 21, 59, 15, 57, 40],
                        [11, 63, 36, 61, 19, 17, 13, 37],
                        [35, 12, 64, 62, 20, 18, 38, 14],
                        [46, 6, 50, 52, 30, 32, 44, 3],
                        [5, 45, 49, 51, 29, 4, 31, 43],
                        [8, 25, 47, 27, 53, 55, 2, 42],
                        [48, 7, 26, 28, 54, 56, 41, 1]])
    return mapping

def butter_filter(sig, cutoff, Fs, btype, order=6,
                  analog=False, axis=-1):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                 category=FutureWarning)
        nyq = 0.5 * Fs
        if btype == 'bandpass':
            normal_low = cutoff.min() / nyq
            normal_high = cutoff.max() / nyq
            if normal_high > 1:
                print('High cutoff frequency too large')
                print('using highpass filter instead')
                btype = 'highpass'
                cutoff = normal_low * nyq
            else:
                b, a = butter(order, (normal_low, normal_high),
                              btype=btype, analog=analog)
        elif (btype == 'highpass') or (btype == 'lowpass'):
            normal = cutoff / nyq
            b, a = butter(order, normal, btype=btype,
                          analog=analog)
        else:
            print('Error, btype must be string of either')
            print('"lowpass", "highpass", or "bandpass"')
            return
        filtered = filtfilt(b, a, sig, axis=axis)
    return filtered

### SThomas  changed 20thDec2022
### based off calibration photos, the new scaling is implemented
def scaleBES(Rpos, R0=1.2, A=0.930, C=1.037):
    scale0 = (A / R0) + C
    scale = (A / Rpos) + C
    return scale / scale0

### renamed to scaleBESOld
def scaleBESOld(Rpos, device='MAST-U', addon=0.24116):
    rBeam = 0.7 # Beam tangency radius, m
    rBeamPort = 2.033 # Beam port radius, m
    rNominal = 1.2 # nominal viewing radius for system
    lNom = 1.527373711104126 # the distance from mirror to R=1.2 location, m, calculated from AF code
    if device == 'MAST-U':
        phiBeamMirrorDeg = 38.3 # angle between beam port and mirror, degrees
        rMirror = 1.868 # major radius of mirror, m
        zMirror = 0.220 # height of mirror, m
    elif device == 'MAST':
        phiBeamMirrorDeg = 30.0 # angle between beam port and mirror, degrees
        rMirror = 1.869 # major radius of mirror, m
        zMirror = 0.395 # height of mirror, m
    ### the beam-mirror angle difference
    phiBeamMirrorRad = np.deg2rad(phiBeamMirrorDeg) # radians
    ### angle between viewed location from tangency point
    alphaViewRad = np.arccos(rBeam / Rpos) # radians
    ### angle between tangency point and beam port
    betaRad = np.arccos(rBeam / rBeamPort) # radians
    ### angle between viewed location and beam port
    gammaRad = betaRad - alphaViewRad # radians
    ### angle between viewed location and mirror
    deltaRad = gammaRad + phiBeamMirrorRad # radians
    ### distance of viewed location from point above mirror in midplane
    dView = np.sqrt(Rpos**2 + rMirror**2 - (2. * Rpos * rMirror * np.cos(deltaRad)))
    ### angle between LoS midplane projection and Rpos radius
    ### absolute angle of view plane anti-cwise from North
    ### angle of elevatrion of LoS above horizontal
    elevRad = np.arctan2(zMirror, dView) # radians
    ### distance of Rpos from mirror
    lView = dView / np.cos(elevRad)
    ### calculating the scale
    # scale = lView  / lNom
    scale = (lView + addon) / (lNom + addon)
    return scale
### end of SThomas change 20thDec2022

### SThomas change 20thDec2022
### changed sep from 0.02 to 0.01812 based of
### calibration photos
def getRz0(Rcentre, ncol=8, nrow=8, sep=sep, chip0=chip0,
            chip1=chip1, chip2=chip2):
    ### empty array for z
    z = np.zeros((ncol, nrow))
    ### z positions based off chip spacing
    z[3,:] = sep / chip0 * chip2 # 1.80e-2
    z[2,:] = z[3,:] + sep # 3.80e-2
    z[1,:] = z[2,:] + (sep / chip0 * chip1) # 6.06e-2
    z[0,:] = z[1,:] + sep # 8.06e-2
    z[4,:] = -sep / chip0 * chip2 # -1.80e-2
    z[5,:] = z[4,:] - sep # -3.80e-2
    z[6,:] = z[5,:] - (sep / chip0 * chip1) # 6.06e-2
    z[7,:] = z[6,:] - sep # -8.06e-2
    ### empty array for R
    R = np.zeros((ncol, nrow))
    ### for iterating over R
    Rm = (nrow - 1) * sep * 0.5
    ### iterate over the R positions
    for i in range(0, nrow):
        ### SThomas change 20thDec2022
        ### flipping in the x direction after checking
        ### calibration photos and the plasma growth
        # R[:,i] = Rcentre + Rm + (i * -sep)
        R[:,i] = Rcentre - Rm + (i * sep)
    return R, z

### SThomas change 3rdJan2023
### added function to give scaled positions from given Rcentre
def getRzCentre(Rcentre, ncol=8, nrow=8, sep=sep, chip0=chip0,
                chip1=chip1, chip2=chip2):
    ### get unadjusted centre positions
    R, z = getRz0(Rcentre, ncol=ncol, nrow=nrow, sep=sep,
                    chip0=chip0, chip1=chip1, chip2=chip2)
    ### for iterating over R
    # Rm = (nrow - 1) * sep * 0.5           ### not needed?
    ### iterate over the R positions
    for i in range(0, nrow):
        scale = scaleBES(R[0,i])
        R[:,i] = ((R[:,i] - Rcentre) * scale + Rcentre)
        z[:,i] *= scale
    return R, z

### SThomas change 20thDec2022
### changed sep from 0.02 to 0.01812 based of
### calibration photos
### SThomas change 3rdJan2023
### part of function taken out for use on its own, then
### ammended this function to call that one
### also ammended to have chip0,1,2 as kwarg inputs
def getRz(shotn, ncol=8, nrow=8, sep=sep, chip0=chip0,
            chip1=chip1, chip2=chip2):
    if shotn < 45177:
        R, z = getRzOld(shotn)
        return R, z
    ### get Rzcentre from list
    _, Rcentre = getInfo(shotn)
    ### call getRzCentre with Rcentre
    R, z = getRzCentre(Rcentre, ncol=ncol, nrow=nrow, sep=sep,
                        chip0=chip0, chip1=chip1, chip2=chip2)
    return R, z

### SThomas change 20thDec2022
### changed sep from 0.02 to 0.01812 based of
### calibration photos
### SThomas change 3rdJan2023
### ammended function to take chip0,1,2 as kwarg inputs
### and also changed chip to chip0
def getRzEdges(Rcentre, ncol=8, nrow=8, sep=sep,
                chip0=chip0, chip1=chip1, chip2=chip2):
    ### calculate number of lines
    nc2 = int(ncol / 4 * 6)
    nr2 = nrow + 1
    ### get unadjusted centre positions
    ### STh 3rdJan23 - changed to take chip kwargs
    Rc, zc = getRz0(Rcentre, ncol=ncol, nrow=nr2, sep=sep,
                    chip0=chip0, chip1=chip1, chip2=chip2)
    ### empty arrays for R and z
    R = np.zeros((nc2, nr2))
    z = np.zeros((nc2, nr2))
    ### for iterating over the z positions
    I = 0
    ### iterate over z
    ### STh 3rdJan23 - changed chip to chip0
    for i in range(0, nc2//3):
        j = (I*2)
        z[(i*3)+0] = zc[I*2].mean() + (sep / chip0 * (chip0 / 2.))
        z[(i*3)+1] = zc[I*2+1].mean() + (sep / chip0 * (chip0 / 2.))
        z[(i*3)+2] = zc[I*2+1].mean() - (sep / chip0 * (chip0 / 2.))
        I += 1
    ### for iterating over R
    # Rm = (nrow - 1) * sep * 0.5           ### not needed?
    ### iterate over the R positions
    for i in range(0, nr2):
        scale = scaleBES(Rc[:,i].mean())
        R[:,i] = ((Rc[:,i].mean() - Rcentre) * scale + Rcentre)
        z[:,i] *= scale
    return R, z

def getRzOld(shotn, splitRz=True):
    if shotn < 45177:
        client = pyuda.Client()
        Rcentre = client.get('devices/d4_mirror/viewRadius',
                             '$MAST_DATA/{}/LATEST/xbt0{}.nc'
                             .format(shotn, shotn)).data[0] # metres
        Rz = np.zeros((4,8,2)) # empty array for R and z, metres
        for i in range(0, Rz.shape[0]):
            for j in range(0, Rz.shape[1]):
                Rz[i,j,0] = (j * -0.02) + 0.07 + Rcentre # metres
                Rz[i,j,1] = (i * 0.02) - 0.03 # metres
    else:
        fname, Rcentre = getInfo(shotn)
        Rz = np.zeros((8,8,2))
        for i in range(0, Rz.shape[0]):
            for j in range(0, Rz.shape[1]):
                Rz[i,j,0] = (j * -0.02) + 0.07 + Rcentre
                Rz[i,j,1] = (i * -0.02) + 0.07
    if splitRz:
        return Rz[:,:,0], Rz[:,:,1]
    else:
        return Rz

def getRzpcolor(shotn, splitRz=True):
    if shotn < 45177:
        client = pyuda.Client()
        Rcentre = client.get('devices/d4_mirror/viewRadius',
                             '$MAST_DATA/{}/LATEST/xbt0{}.nc'
                             .format(shotn, shotn)).data[0] # metres
        Rz = np.zeros((5,9,2)) # empty array for R and z, metres
        for i in range(0, Rz.shape[0]):
            for j in range(0, Rz.shape[1]):
                Rz[i,j,0] = (j * -0.02) + 0.08 + Rcentre # metres
                Rz[i,j,1] = (i * 0.02) - 0.04 # metres
    else:
        fname, Rcentre = getInfo(shotn)
        Rz = np.zeros((9,9,2))
        for i in range(0, Rz.shape[0]):
            for j in range(0, Rz.shape[1]):
                Rz[i,j,0] = (j * -0.02) + 0.08 + Rcentre
                Rz[i,j,1] = (i * -0.02) + 0.08
    if splitRz:
        return Rz[:,:,0], Rz[:,:,1]
    else:
        return Rz

def find_nearest(arr, val):
    return np.abs(arr - val).argmin()













































#
