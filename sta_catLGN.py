# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:33:26 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.ndimage
from matplotlib import gridspec
from tqdm import tqdm

# Load data
data = scipy.io.loadmat('c2p3.mat')
 
# stim : 16 x 16 images that were presented at the corresponding times.
stim = data['stim'] # (16, 16, 32767)

# counts : vector containing the number of spikes in each 15.6 ms bin.
counts = data['counts']
counts = np.reshape(counts, (-1)) # (32767)

"""
Calculate the spike-triggered average images 
for each of the 12 time steps(= 187.2 ms) before each spike
"""
spike_timing = np.where(counts > 0)[0]
num_spikes = np.sum(counts > 0) # number of spikes

num_timesteps = 12
H, W = 16, 16 # height and width of stimulus image 
sta = np.zeros((H, W, num_timesteps)) #spike-triggered average

for t in spike_timing:
    if t > num_timesteps:
        sta += stim[:, :, t-num_timesteps:t]*counts[t]
   
sta /= num_spikes

"""
Plot results
"""
for T in tqdm(range(num_timesteps)):
    fig = plt.figure(figsize=(10,3))
    fig.suptitle("The receptive field of a cat LGN X cell. (timesteps : "+str(T)+")", fontsize=14)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2], height_ratios=[1]) 
    
    x = np.arange(-W//2, W//2+2, 2)
    y = np.arange(-H//2, H//2+2, 2)
    
    ax1 = plt.subplot(gs[0,0])
    plt.imshow(sta[:,:,T])
    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.xlabel('x')
    plt.ylabel('y')
    ax1.title.set_text('heatmap')
    plt.xticks(list(np.arange(0, W+1, 2)-0.5), list(x))
    plt.yticks(list(np.arange(0, H+1, 2)-0.5), list(y))
    
    
    zoom = 3
    smoothed_sta = scipy.ndimage.zoom(sta[:,:,T], zoom=zoom) # smoothed
    
    x = np.arange(-W//2*zoom, W//2*zoom)
    y = np.arange(-H//2*zoom, H//2*zoom)
    X, Y = np.meshgrid(x, y)
    
    ax2 = plt.subplot(gs[0,1])
    plt.contour(X, Y, smoothed_sta, 25)
    plt.xlabel('x')
    #plt.ylabel('y')
    ax2.title.set_text('contour')
    plt.gca().set_aspect('equal')
    
    ax3 = plt.subplot(gs[0,2], projection='3d')
    ax3.plot_surface(X, Y, smoothed_sta, shade=True,
                     color='grey')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    #plt.show()
    plt.savefig("RF_catLGN_time"+str(T)+".png")
    plt.close()

# Calculate spatio-temporal receptive field
sum_sta = np.sum(sta, axis=1).T

fig = plt.figure(figsize=(12, 3))
fig.suptitle("The spatio-temporal receptive field of a cat LGN X cell.", fontsize=14)
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.5],
                       height_ratios=[1]) 

x = np.arange(-H//2, H//2+2, 2)

ax1 = plt.subplot(gs[0,0])
plt.imshow(sum_sta)
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()
plt.xlabel('x')
plt.ylabel('time steps')
ax1.title.set_text('heatmap')
plt.xticks(list(np.arange(0, W+1, 2)-0.5), list(x))

zoom = 3
smoothed_sum_sta = scipy.ndimage.zoom(sum_sta, zoom=zoom) # smoothed

x = np.arange(-H//2*zoom, H//2*zoom)
t = np.arange(12*zoom)
X, T = np.meshgrid(x, t)

ax2 = plt.subplot(gs[0,1])
plt.contour(X, T, smoothed_sum_sta, 25)
plt.xlabel('x')
#plt.ylabel('y')
ax2.title.set_text('contour')
plt.gca().set_aspect('equal')

ax3 = plt.subplot(gs[0,2], projection='3d')
ax3.plot_surface(X, T, smoothed_sum_sta, shade=True,
                 color='grey')
ax3.set_xlabel('x')
ax3.set_ylabel('time steps')
#plt.show()
plt.savefig("RF_catLGN_spattemp.png")
plt.close()