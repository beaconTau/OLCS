#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on April 29 2020

@author: romerowo, swissel
"""

import numpy as np

class Detector_Array:
    def __init__(self, mode='prototype_2019'):
        if mode=='prototype_2018':
            # position of antenna0 measured in Google Earth for 2018 array. Should be updated
            self.detector_altitude_km = 3.87553 # Elevation of center of antenna 0 in km
            self.detector_lat_A0 = 37.589310    # Latitude in degrees of A0
            self.detector_lon_A0 = -118.237621  # Longitude in degrees of A0
            # stored in km, using estimate from Kaeli Hughes of the antenna positions as of 4/29/2020
            self.x = np.array([0,-6.039, -1.272, 3.411 ]) * 1e-3
            self.y = np.array([0,-1.618, -10.362, -11.897]) * 1e-3
            self.z = np.array([0,2.275, 1.282, -0.432]) * 1e-3 + self.detector_altitude_km
            
        if mode=='prototype_2019':  
            # position of antenna0 measured in Google Earth
            self.detector_altitude_km = 3.87553 # Elevation of center of antenna 0 in km
            self.detector_lat_A0 = 37.589310    # Latitude in degrees of A0
            self.detector_lon_A0 = -118.237621  # Longitude in degrees of A0
            # stored in km, using estimate from Dan Southall of the physical positions of the antennas as of 4/29/2020
            self.x = np.array([0,-33.49373061801294, -8.660668526919165, -32.16822443711699 ]) * 1e-3
            self.y = np.array([0,-12.216161215847706, -44.5336329143579, -43.200941168610264]) * 1e-3
            self.z = np.array([0,15.23990049241125, 5.489838290379097, 11.889772376808601]) * 1e-3 + self.detector_altitude_km  
        
    ####################################################################################

    def get_distances_and_view_angles(self, XmaxC, Geom, event=0):
        self.th_view   = np.zeros(len(self.x))
        self.dist_Xmax = np.zeros(len(self.x))
        dx1 = XmaxC.x_max[event] - Geom.x_pos[event]
        dy1 = XmaxC.y_max[event] - Geom.y_pos[event]
        dz1 = (XmaxC.z_max[event]- XmaxC.Earth_radius) - Geom.z_pos[event]
        r1 = np.sqrt( dx1**2 + dy1**2 + dz1**2 )
        dx2 = XmaxC.x_max[event] - self.x
        dy2 = XmaxC.y_max[event] - self.y
        dz2 = (XmaxC.z_max[event]- XmaxC.Earth_radius) - self.z
        r2  = np.sqrt( dx2**2 + dy2**2 + dz2**2 )
        dx3 = self.x - Geom.x_pos[event]
        dy3 = self.y - Geom.y_pos[event]
        dz3 = self.z - Geom.z_pos[event]
        r3 = np.sqrt( dx3**2 + dy3**2 + dz3**2 )
        #cos_view_angle    = (r1**2 + r2**2 - r3**2) / (2.*r1*r2)
        cos_view_angle = (dx1*dx2 + dy1*dy2 + dz1*dz2) / (r1*r2)
        #print cos_view_angle
        self.th_view   = np.arccos(cos_view_angle)
        self.dist_Xmax = np.array(r2) 
    
    def plot_array():
        
