# SR-figure-code
# Python 3.7 - code for maps and figures in manuscript "Sea ice variability and maritime activity around Svalbard in the period 2012-2019"
# figure 1: lines XYZ - XYZ
# figure 2: lines XYZ - XYZ
# figure 3: lines XYZ - XYZ
# figure 4: lines XYZ - XYZ
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
libraries used for all functions in this file 

from PIL import Image
from numpy.random import rand
import matplotlib.gridspec as gridspec
from matplotlib.cbook import get_sample_data
from matplotlib.colors import LightSource
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import math
import mpl_toolkits.basemap as basemap
import cmocean
import matplotlib.patches as mpatches
import matplotlib.patheffects as PE
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np

Polar Stereographic projection
function retieved from: http://code.activestate.com/recipes/578379-plotting-maps-with-polar-stereographic-projection-/

def polar_stere(lon_w, lon_e, lat_s, lat_n, **kwargs):
    '''Returns a Basemap object (NPS/SPS) focused in a region.

    lon_w, lon_e, lat_s, lat_n -- Graphic limits in geographical coordinates.
                                  W and S directions are negative.
    **kwargs -- Aditional arguments for Basemap object.

    '''
    lon_0 = lon_w + (lon_e - lon_w) / 2.
    ref = lat_s if abs(lat_s) > abs(lat_n) else lat_n
    lat_0 = math.copysign(90., ref)
    proj = 'npstere' if lat_0 > 0 else 'spstere'
    prj = basemap.Basemap(projection=proj, lon_0=lon_0, lat_0=lat_0,
                          boundinglat=0, resolution='c')
    lons = [lon_w, lon_e, lon_w, lon_e, lon_0, lon_0]
    lats = [lat_s, lat_s, lat_n, lat_n, lat_s, lat_n]
    x, y = prj(lons, lats)
    ll_lon, ll_lat = prj(min(x), min(y), inverse=True)
    ur_lon, ur_lat = prj(max(x), max(y), inverse=True)
    return basemap.Basemap(projection='stere', lat_0=lat_0, lon_0=lon_0,
                           llcrnrlon=ll_lon, llcrnrlat=ll_lat,
                           urcrnrlon=ur_lon, urcrnrlat=ur_lat, **kwargs)
    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# figure 1a) - map of Svalbard with settlements 

def Svalbard():
    fig, ax1 = plt.subplots(figsize=(8, 10), dpi=300)  # (12,10), dpi = 300)
    zoom_scale = 0
    lat_s = 72.0
    lat_n = 84.4
    lon_w = 2
    lon_e = 38.0
    # resolution crude (c), low (l), intermediate (i), high (h), full (f)
    m = polar_stere(lon_w, lon_e, lat_s, lat_n, resolution='f')
    m.drawcoastlines(linewidth=0.5)
    m.drawparallels(np.arange(0, 90, 2), labels=[
                    1, 0, 0, 0], color='darkgrey', fontsize=11, rotation=90)
    m.drawmeridians(np.arange(0, 360, 10), labels=[
                    0, 0, 0, 1], color='darkgrey', fontsize=11)
    m.fillcontinents(color='grey', lake_color='grey')
    m.drawmapboundary(fill_color='#f5f5f5')
    # a workaround to get meridians plotted
    phs = np.arange(80, 90, 2.0)
    for ea in np.arange(0, 360, 10.0):
        lds = np.ones(len(phs))*ea
        m.plot(lds, phs, latlon=True, color="darkgrey",
               linewidth=1, dashes=[1, 1])
    Longylat, Longylon = 78.2232, 15.6267
    xl, yl = m(Longylon, Longylat)
    xl2, yl2 = m(7, 77.55)
    plt.plot(xl, yl, 'k*', markersize=10)
    lg = plt.annotate('Longyearbyen', xy=(xl, yl), xycoords='data', xytext=(
        xl2, yl2), textcoords='data', color='k', fontsize=13, fontweight='bold')
    lg.set_path_effects([PE.withStroke(linewidth=1, foreground='whitesmoke')])
    plt.draw()
    Islat, Islon = 78, 8
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(5, 78.05)
    plt.annotate('Isfjorden', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=11, fontstyle='italic', fontweight='roman', rotation=0)
    lat = np.linspace(78.25, 78.15)
    lon = np.linspace(9.8, 14)
    x1, y1 = m(lon, lat)
    m.plot(x1, y1, linewidth=1, color='k')
    Storenlat, Storenlon = 77.81, 19.66
    xS, yS = m(Storenlon, Storenlat)
    xS2, yS2 = m(18.35, 76.9)
    plt.annotate('Storfjorden', xy=(xS, yS), xycoords='data', xytext=(xS2, yS2), textcoords='data',
                 color='k', fontsize=11, fontstyle='italic', fontweight='roman', rotation=80)
    Storenalat, Storenalon = 76.26, 17.88
    xS, yS = m(Storenalon, Storenalat)
    xS2, yS2 = m(16.3, 76.2)
    plt.annotate('Storfjordrenna', xy=(xS, yS), xycoords='data', xytext=(
        xS2, yS2), textcoords='data', color='k', fontsize=11, fontstyle='italic', fontweight='roman')
    Hinlat, Hinlon = 79.95, 17.58
    xh, yh = m(Hinlon, Hinlat)
    xh2, yh2 = m(22.0, 78.9)
    plt.annotate('Hinlopen Strait', xy=(xh, yh), xycoords='data', xytext=(xh2, yh2), textcoords='data',
                 color='k', fontsize=11, fontstyle='italic', fontweight='roman', rotation=15)
    lat = np.linspace(79.6, 79.0)
    lon = np.linspace(18.8, 22.0)
    x1, y1 = m(lon, lat)
    m.plot(x1, y1, linewidth=1, color='k')
    Maglat, Maglon = 79.58, 10.76
    xm, ym = m(Maglon, Maglat)
    xm2, ym2 = m(0, 79.33)
    mg = plt.annotate('Magdalenefjorden', xy=(xm, ym), xycoords='data', xytext=(
        xm2, ym2), textcoords='data', color='k', fontsize=11, fontstyle='italic', fontweight='roman')
    lat = np.linspace(79.7, 79.57)
    lon = np.linspace(8, 11)
    x1, y1 = m(lon, lat)
    m.plot(x1, y1, linewidth=1, color='k')
    Konlat, Konlon = 78.95, 12.14
    xk, yk = m(Konlon, Konlat)
    xk2, yk2 = m(3, 78.75)
    mg = plt.annotate('Kongsfjorden', xy=(xk, yk), xycoords='data', xytext=(
        xk2, yk2), textcoords='data', color='k', fontsize=11, fontstyle='italic', fontweight='roman')
    lat = np.linspace(79.08, 78.95)
    lon = np.linspace(10.4, 12)
    x1, y1 = m(lon, lat)
    m.plot(x1, y1, linewidth=1, color='k')
    SFPZ = [[34.1, 83.7], [28.7, 84.0], [7.84, 83.69], [5.84, 82.45], [-1.41, 79.86], [-3.34, 78.32], [-2.7, 76.89], [0.89, 75.97], [6.38, 74.31], [10.3, 72.2],
            [16.57, 73.58], [19.9, 74.25], [26.08, 74.49], [30.84, 74.34], [
                33.81, 73.98], [36.98, 74.93], [36.98, 75.77], [37.99, 78.63],
            [34.83, 79.3], [35.11, 83.33], [34.1, 83.7]]  # last on 12.12.19
    for coord_pair in SFPZ:
        coord_pair[0], coord_pair[1] = m(coord_pair[0], coord_pair[1])
    poly = Polygon(SFPZ, fill=None, edgecolor='royalblue',
                   linewidth=2)  # facecolor='lightgreen'
    ax1.add_patch(poly)
    m.drawmapscale(34, 72.2, 44, 73.2, 200, units='km', fontsize=11, yoffset=None, barstyle='fancy',
                   labelstyle='simple', fillcolor1='white', fillcolor2='dimgrey', fontcolor='dimgrey', zorder=7)
    plt.tight_layout()
    plt.savefig('/Users/ali/Desktop/Svalbard2705.png', bbox_inches='tight',
                pad_inches=0.05)
    plt.show()

Svalbard() 

# figure 1b) - map of geographic location in the Arctic

def Arctic():
    fig, ax1 = plt.subplots(figsize=(12, 12), dpi=300)
    m = basemap.Basemap(projection='npstere', boundinglat=62,
                        lon_0=0, resolution='f')
    # m.bluemarble()
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='grey', lake_color='grey')
    # draw parallels and meridians.
    # m.drawparallels(np.arange(0., 81., 10.))
    # m.drawmeridians(np.arange(-180., 181., 20.))
    m.drawmapboundary(fill_color='#f5f5f5')
    # draw tissot's indicatrix to show distortion.
    # plt.title("North Polar Stereographic Projection")
    square = [[2, 73], [-18, 82], [58, 82], [36, 73], [2, 73]]
    for coord_pair in square:
        coord_pair[0], coord_pair[1] = m(coord_pair[0], coord_pair[1])
    poly = Polygon(square, fill=None, edgecolor='red',
                   linewidth=16)  # facecolor='lightgreen'
    ax1.add_patch(poly)
    plt.tight_layout()
    plt.savefig('/Users/ali/Desktop/Arctic_dpi300.png')
    # plt.show()

Arctic()

def multifig1():
    plt.figure(figsize=(13, 14), dpi=300)
    gs1 = gridspec.GridSpec(3, 3)
    gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

    ax1 = plt.subplot(gs1[0:, :-1])
    plt.axis('off')
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.text(0, 0.9, 'a)', fontsize=16, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax1.transAxes)
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/Svalbard2705.png'))
    plt.imshow(im)

    ax1 = plt.subplot(gs1[0, -1])
    plt.axis('on')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.text(0, 0.9, 'b)', fontsize=16, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/Arctic_dpi300.png'))
    plt.imshow(im)

    plt.savefig('/Users/ali/Desktop/Figure1_dpi300.png', bbox_inches='tight',
                pad_inches=0.05)
    plt.show()


multifig1()

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

# figure
