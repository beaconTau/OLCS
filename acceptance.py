#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:25:21 2018

@author: romerowo
"""

# import matplotlib
# matplotlib.use('Agg')
import sys
from pylab import *

from geometry import Geometry
from xmax_calc import Xmax_calc
from detector_array import Detector_Array
from radio_emission import Radio_Emission
from detector import Detector

def acceptance_sim(radius_km = 10., 
                   num_particles = 10000, 
                   zenith_angle_deg = None, 
                   log10_energy=17.,
                   SNR_thresh = 5.,
                   N_trig = 1,
                   trig_pol = 'hpol',
                   detector_mode = 'prototype_2019',
                   detector_altitude_km = 3.87553,
                   verbose = False):
    '''
    SETUP
    '''
    geom = Geometry(radius_km, num_particles, detector_altitude_km, zenith_angle_deg)
    
    XMC = Xmax_calc()
    # I want to provide the energy, cosmic ray ground position and direction, and get the Xmax position
    XMC.get_Xmax_position(log10_energy, geom)
    Xmax_altitude = np.sqrt(XMC.x_max**2 + XMC.y_max**2 + XMC.z_max**2) - XMC.Earth_radius - XMC.detector_altitude_km
    
    det_arr = Detector_Array(mode=detector_mode)
    
    rad_em = Radio_Emission(plots=False)
    
    if detector_mode == 'prototype_2019':
        antenna_type = 'dipole_2019'
    elif detector_mode == 'prototype_2018':
        antenna_type = 'lwa_2018'
    else:
        raise ValueError("Valid detector types are 'prototype_2019' and 'prototype_2018'. You requested '%s'"%detector_type)
    det = Detector(antenna_type=antenna_type)
    
    # loop through simulated events:
    trigger = 0
    trig_evn = []
    thz_array = []
    xmax_altitude_array = []
    view_angle_array = []

    for evn in range(0, num_particles): 
        if evn%500 == 0: 
            if verbose: 
                print(('%d of %d'%(evn, num_particles)))
    
        det_arr.get_distances_and_view_angles(XMC, geom, event=evn)
        # first cut by minimum view angle here.
        if np.min(det_arr.th_view*180./pi)> 10.: continue
    
        ph_xy = geom.phi_CR[evn]*180./pi
        th_z = np.arccos(geom.k_z[evn])*180./pi
        E_field,dist = rad_em.radio_beam_model2(th_z, det_arr.th_view, XMC.detector_altitude_km) 
        E_field *= 10**(log10_energy-17.)
        x_pol, y_pol, z_pol = rad_em.get_pol(XMC.x_max[evn],  XMC.y_max[evn],  XMC.z_max[evn]-XMC.Earth_radius, geom.x_pos[evn], geom.y_pos[evn], geom.z_pos[evn])
        max_val = np.max(np.array(E_field)*1.e6)
        # then cut by maximum voltage here
        
        V_x = det.Efield_2_Voltage(np.array(E_field*x_pol), th_z, ph_xy)
        V_y = det.Efield_2_Voltage(np.array(E_field*y_pol), th_z, ph_xy)
        V_z = det.Efield_2_Voltage(np.array(E_field*z_pol), th_z, ph_xy)
    
        V_mag = np.sqrt(V_x**2 + V_y**2 + V_z**2)
    
        max_val_V = np.max(V_mag)*1.e6
        
        #SNR_x = np.abs(V_x + np.random.normal(0., det.V_rms, len(V_x)))/det.V_rms
        #SNR_y = np.abs(V_y + np.random.normal(0., det.V_rms, len(V_x)))/det.V_rms
        #SNR_z = np.abs(V_z + np.random.normal(0., det.V_rms, len(V_x)))/det.V_rms
        
        #require that xmax is above the detector
        cut_xmax = Xmax_altitude[evn] > detector_altitude_km
        
        #require that the azimuthal direction is within 120 deg in azimuth
        cut_phi = np.logical_or( ph_xy < 60., ph_xy > 300.) 
        
        SNR_x = np.abs(V_x)/det.V_rms
        SNR_y = np.abs(V_y)/det.V_rms
        SNR_z = np.abs(V_z)/det.V_rms
    
        cut_x = SNR_x>SNR_thresh
        cut_y = SNR_y>SNR_thresh
        cut_z = SNR_z>SNR_thresh
        
        if( trig_pol == 'hpol'):
            cut_trig_pol = np.logical_or(cut_x, cut_y)
        elif( trig_pol == 'vpol'):
            cut_trig_pol = cut_z
        else:
            print("Warning: trigger must be 'hpol' or 'vpol'. Nothing will trigger")
            cut_trig_pol = np.zeros(len(cut_x))
        
        cut_trig = np.sum(cut_trig_pol) >= N_trig
        cut_event = np.logical_and(cut_xmax, cut_trig)
        cut_event = np.logical_and(cut_event, cut_phi)

        event_trigger = 0
        if  cut_event:
            event_trigger = 1
            thz_array.append(th_z)
            trig_evn.append(evn)
            xmax_altitude_array.append(Xmax_altitude[evn])
            view_angle_array.append(det_arr.th_view*180./np.pi)
            if verbose:
                print(('\t d_core  %1.2f'%(np.sqrt(geom.x_pos[evn]**2 + geom.y_pos[evn]**2))))
                print(('\t th_view %1.2f %1.2f'%(np.min(det_arr.th_view)*180./pi, np.max(det_arr.th_view)*180./pi))) 
                print(('\t th_zen %1.2f'%(np.arccos(geom.k_z[evn])*180./pi)))
                print('\t=================\n')
                
        trigger += event_trigger
        # will want to save events
        #print '\t %1.2e %1.2e %1.2e'%(x_pol, y_pol, z_pol)
        # then cute by maximum V_mag
        #plot([np.min(det_arr.th_view)*180./pi], [max_val], 'k.')
        #semilogy([np.min(det_arr.th_view)*180./pi], [max([max(SNR_x), max(SNR_y), max(SNR_z)])], 'k.')
        
    
    #show()
    #print 'trigger', trigger
    return float(trigger)/float(num_particles), np.array(thz_array), np.array(xmax_altitude_array), np.array(view_angle_array)
    #print 10**(log10_energy-17.)
    #print 'det.V_rms', det.V_rms
    '''
    figure(figsize=(4,4))
    plot(geom.x_pos, geom.y_pos, '.', alpha=0.1)
    xlabel('x, km')
    ylabel('y, km')
    figure()
    hist(geom.x_pos)
    figure()
    hist(geom.th_CR*180./pi, bins=np.arange(0.,91., 5.))
    xticks(np.arange(0.,91., 10.))
    xlabel('CR Zenith Angle, deg')
    show()
    '''
        