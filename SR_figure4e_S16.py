# %%
# code for figure 4e) and Supplementary S16 - map of Svalbard with delimitation of 12 zones

import mpl_toolkits.basemap as basemap
import math
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap, shiftgrid, cm


# Polar Stereographic projection
# function retieved from: http: // code.activestate.com/recipes/578379-plotting-maps-with-polar-stereographic-projection-/

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


arearows = [
    [1, 8.0, 35.0, 81.0, 82.8],
    [2, 8.0, 20.0, 79.8, 81.0],
    [3, 20.0, 36.0, 79.8, 81.0],
    [4, 8.0, 13.0, 78.0, 79.79],
    [5, 17.5, 24.5, 79.0, 79.79],
    [6, 24.51, 36.0, 78.5, 79.79],
    [7, 8.0, 17.0, 76.0, 77.99],
    [8, 17.41, 24.5, 77.11, 79.0],
    [9, 24.51, 38.0, 76.5, 78.49],
    [10, 17.01, 24.5, 76.0, 77.1],
    [11, 8.0, 24.5, 74.0, 76.0],
    [12, 24.5, 38.5, 74.0, 76.5]
]


def areas_rec():
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    lat_s = 73.2
    lat_n = 83.0
    lon_w = 2
    lon_e = 38.0  # 37.0
    m = polar_stere(lon_w, lon_e, lat_s, lat_n, resolution='f')
    m.drawcoastlines()
    m.drawparallels(np.arange(0, 90, 4), labels=[
                    1, 0, 0, 0], color='darkgrey', fontsize=30, rotation=90)
    m.drawmeridians(np.arange(0, 360, 10), labels=[
                    0, 0, 0, 1], color='darkgrey', fontsize=30)
    m.fillcontinents(color='grey', lake_color='grey')
    m.drawmapboundary(fill_color='#f5f5f5')  # 'paleturquoise')#'#16162E')
    # a workaround to get meridians plotted =>  but this only a few between 80 and 82 degrees north...
    phs = np.arange(80, 85, 2.0)
    for ea in np.arange(0, 360, 10.0):
        lds = np.ones(len(phs))*ea
        m.plot(lds, phs, latlon=True, color="darkgrey",
               linewidth=1, dashes=[1, 1])
    """
    # all 12 zones numbered -> supplementary material
    Islat, Islon = 82, 20
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(20, 81.8)
    plt.annotate('1', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 80, 14
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(14, 80.2)
    plt.annotate('2', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 80, 28
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(28, 80.2)
    plt.annotate('3', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 79, 10
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(10, 78.5)
    plt.annotate('4', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 79, 20
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(20, 79.2)
    plt.annotate('5', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 79, 30
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(30, 79)
    plt.annotate('6', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 77, 12
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(12, 76.8)
    plt.annotate('7', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 78, 19
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(19, 77.7)
    plt.annotate('8', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 77.5, 30
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(30, 77.5)
    plt.annotate('9', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 76.5, 20
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(20, 76.5)
    plt.annotate('10', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 75, 15
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(15, 75)
    plt.annotate('11', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    Islat, Islon = 75, 30
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(30, 75)
    plt.annotate('12', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=30, fontweight='bold')
    """
    # zones determined by a,b,c,d for SR paper
    Islat, Islon = 80, 14
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(12, 80.1)
    plt.annotate('a', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=78, fontweight='bold')
    Islat, Islon = 78, 19
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(18.5, 77.5)
    plt.annotate('b', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=78, fontweight='bold')
    Islat, Islon = 79, 30
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(29.5, 79)
    plt.annotate('c', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=76, fontweight='bold')
    Islat, Islon = 76.5, 20
    xI, yI = m(Islon, Islat)
    xI2, yI2 = m(19, 76.1)
    plt.annotate('d', xy=(xI, yI), xycoords='data', xytext=(xI2, yI2), textcoords='data',
                 color='k', fontsize=76, fontweight='bold')

    def area(color, l):
        # top line
        lat = np.linspace((row[4]), (row[4]))
        lon = np.linspace(row[1], (row[2]-0.3))
        x1, y1 = m(lon, lat)
        m.plot(x1, y1, linewidth=l, color=color)
        # bottom line
        lat = np.linspace(row[3], row[3])
        lon = np.linspace(row[1], (row[2]))
        x1, y1 = m(lon, lat)
        m.plot(x1, y1, linewidth=l, color=color)
        # left line
        lat = np.linspace(row[3], (row[4]))
        lon = np.linspace(row[1], row[1])
        x1, y1 = m(lon, lat)
        m.plot(x1, y1, linewidth=l, color=color)
        # right line
        lat = np.linspace(row[3], (row[4]))
        lon = np.linspace((row[2]), (row[2]))
        x1, y1 = m(lon, lat)
        m.plot(x1, y1, linewidth=l, color=color)
    for row in arearows:
        area('#045a8d', 6)
    m.drawmapscale(5.5, 73.4, 10.1, 73.8, 200, units='km', fontsize=24, yoffset=None, barstyle='fancy',
                   labelstyle='simple', fillcolor1='grey', fillcolor2='white', fontcolor='grey', zorder=5)
    textstr = '    \n  \n'
    props = dict(boxstyle='round', facecolor='#045a8d', edgecolor='#045a8d',
                 alpha=0.7)
    ax.text(0.374, 0.648, textstr, transform=ax.transAxes,
            fontsize=32, verticalalignment='top', rotation=-5.5, bbox=props)
    plt.tight_layout()
    plt.savefig('/Users/ali/Desktop/areas_dpi300.png')
    plt.show()


areas_rec()

# %%
