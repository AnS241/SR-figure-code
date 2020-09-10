# %%
# code for the creation of maps in figure 3 and Supplementary figures S1 to S15

import matplotlib.patches as mpatches
import cmocean
import mpl_toolkits.basemap as basemap
import math
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from matplotlib.cbook import get_sample_data
import matplotlib.gridspec as gridspec
from PIL import Image
import glob


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

# figure 3 and S1 - S15


os.chdir('/Volumes/AS_data/IceData')

# list of sea ice concentration files
ice_2012 = ['IUP_ASI_201208_svalbard_monthly.txt', 'IUP_ASI_201209_svalbard_monthly.txt',
            'IUP_ASI_201210_svalbard_monthly.txt', 'IUP_ASI_201211_svalbard_monthly.txt', 'IUP_ASI_201212_svalbard_monthly.txt']
ice_2013 = ['IUP_ASI_201301_svalbard_monthly.txt', 'IUP_ASI_201302_svalbard_monthly.txt', 'IUP_ASI_201303_svalbard_monthly.txt', 'IUP_ASI_201304_svalbard_monthly.txt', 'IUP_ASI_201305_svalbard_monthly.txt', 'IUP_ASI_201306_svalbard_monthly.txt',
            'IUP_ASI_201307_svalbard_monthly.txt', 'IUP_ASI_201308_svalbard_monthly.txt', 'IUP_ASI_201309_svalbard_monthly.txt', 'IUP_ASI_201310_svalbard_monthly.txt', 'IUP_ASI_201311_svalbard_monthly.txt', 'IUP_ASI_201312_svalbard_monthly.txt']
ice_2014 = ['IUP_ASI_201401_svalbard_monthly.txt', 'IUP_ASI_201402_svalbard_monthly.txt', 'IUP_ASI_201403_svalbard_monthly.txt', 'IUP_ASI_201404_svalbard_monthly.txt', 'IUP_ASI_201405_svalbard_monthly.txt', 'IUP_ASI_201406_svalbard_monthly.txt',
            'IUP_ASI_201407_svalbard_monthly.txt', 'IUP_ASI_201408_svalbard_monthly.txt', 'IUP_ASI_201409_svalbard_monthly.txt', 'IUP_ASI_201410_svalbard_monthly.txt', 'IUP_ASI_201411_svalbard_monthly.txt', 'IUP_ASI_201412_svalbard_monthly.txt']
ice_2015 = ['IUP_ASI_201501_svalbard_monthly.txt', 'IUP_ASI_201502_svalbard_monthly.txt', 'IUP_ASI_201503_svalbard_monthly.txt', 'IUP_ASI_201504_svalbard_monthly.txt', 'IUP_ASI_201505_svalbard_monthly.txt', 'IUP_ASI_201506_svalbard_monthly.txt',
            'IUP_ASI_201507_svalbard_monthly.txt', 'IUP_ASI_201508_svalbard_monthly.txt', 'IUP_ASI_201509_svalbard_monthly.txt', 'IUP_ASI_201510_svalbard_monthly.txt', 'IUP_ASI_201511_svalbard_monthly.txt', 'IUP_ASI_201512_svalbard_monthly.txt']
ice_2016 = ['IUP_ASI_201601_svalbard_monthly.txt', 'IUP_ASI_201602_svalbard_monthly.txt', 'IUP_ASI_201603_svalbard_monthly.txt', 'IUP_ASI_201604_svalbard_monthly.txt', 'IUP_ASI_201605_svalbard_monthly.txt', 'IUP_ASI_201606_svalbard_monthly.txt',
            'IUP_ASI_201607_svalbard_monthly.txt', 'IUP_ASI_201608_svalbard_monthly.txt', 'IUP_ASI_201609_svalbard_monthly.txt', 'IUP_ASI_201610_svalbard_monthly.txt', 'IUP_ASI_201611_svalbard_monthly.txt', 'IUP_ASI_201612_svalbard_monthly.txt']
ice_2017 = ['IUP_ASI_201701_svalbard_monthly.txt', 'IUP_ASI_201702_svalbard_monthly.txt', 'IUP_ASI_201703_svalbard_monthly.txt', 'IUP_ASI_201704_svalbard_monthly.txt', 'IUP_ASI_201705_svalbard_monthly.txt', 'IUP_ASI_201706_svalbard_monthly.txt',
            'IUP_ASI_201707_svalbard_monthly.txt', 'IUP_ASI_201708_svalbard_monthly.txt', 'IUP_ASI_201709_svalbard_monthly.txt', 'IUP_ASI_201710_svalbard_monthly.txt', 'IUP_ASI_201711_svalbard_monthly.txt', 'IUP_ASI_201712_svalbard_monthly.txt']
ice_2018 = ['IUP_ASI_201801_svalbard_monthly.txt', 'IUP_ASI_201802_svalbard_monthly.txt', 'IUP_ASI_201803_svalbard_monthly.txt', 'IUP_ASI_201804_svalbard_monthly.txt', 'IUP_ASI_201805_svalbard_monthly.txt', 'IUP_ASI_201806_svalbard_monthly.txt',
            'IUP_ASI_201807_svalbard_monthly.txt', 'IUP_ASI_201808_svalbard_monthly.txt', 'IUP_ASI_201809_svalbard_monthly.txt', 'IUP_ASI_201810_svalbard_monthly.txt', 'IUP_ASI_201811_svalbard_monthly.txt', 'IUP_ASI_201812_svalbard_monthly.txt']
ice_2019 = ['IUP_ASI_201901_svalbard_monthly.txt', 'IUP_ASI_201902_svalbard_monthly.txt', 'IUP_ASI_201903_svalbard_monthly.txt', 'IUP_ASI_201904_svalbard_monthly.txt',
            'IUP_ASI_201905_svalbard_monthly.txt', 'IUP_ASI_201906_svalbard_monthly.txt', 'IUP_ASI_201907_svalbard_monthly.txt', 'IUP_ASI_201908_svalbard_monthly.txt', 'IUP_ASI_201909_svalbard_monthly.txt']


def freqmaps():
    icefile = ice_2017
    shipfile = sorted(
        glob.glob('/Users/ali/Desktop/ships/STcut/ais_ST2017*.csv'))
    # for 2012 [8,9,10,11,12] / for 2019 [1,2,3,4,5,6,7,8,9]
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # for 2012 ['August','September','October','November','December'] /for 2019 ['January','February','March','April','May','June','July','August','September']
    text = ['January', 'February', 'March', 'April',
            'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    for i, s, BLA, t in zip(icefile, shipfile, xs, text):
        fig, ax1 = plt.subplots(figsize=(12, 10), dpi=300)
        lat_s = 73.2
        lat_n = 83.0
        lon_w = 2
        lon_e = 38.0
        # resolution crude (c), low (l), intermediate (i), high (h), full (f)
        m = polar_stere(lon_w, lon_e, lat_s, lat_n, resolution='i')
        m.drawcoastlines()
        m.drawparallels(np.arange(0, 90, 2), labels=[
                        1, 0, 0, 0], color='darkgrey', fontsize=20, rotation=90)
        m.drawmeridians(np.arange(0, 360, 5), labels=[
                        0, 0, 0, 1], color='darkgrey', fontsize=20)
        m.fillcontinents(color='grey', lake_color='grey')  # 'tan'
        m.drawmapboundary(fill_color='aliceblue')
        # a workaround to get meridians plotted in polar regions, one of the main defaults of basemap
        phs = np.arange(80, 85, 2.0)
        for ea in np.arange(0, 360, 5.0):
            lds = np.ones(len(phs))*ea
            m.plot(lds, phs, latlon=True, color="darkgrey",
                   linewidth=0.8, dashes=[1, 1])
    # add sea ice concentration
        ICE = np.loadtxt(i, dtype=float, delimiter=',')
        LAT = np.loadtxt('latM.txt', dtype=float, delimiter=',')
        LON = np.loadtxt('lonM.txt', dtype=float, delimiter=',')
        x, y = m(LON, LAT)
        cmap = cmocean.cm.ice
        concentration = [0, 15, 30, 45, 60, 75, 90, 100]
        cs = m.contourf(x, y, ICE, concentration, cmap=cmap)
    # add vessel positions
        # first select rows of interest, and specify that the column date_time_utc is datetime dtype.
        aisdata = pd.read_csv(s, delimiter=',', dtype={
                              'imo_nr': str, 'length': str})
        # print(aisdata.head())
        aisdata['imo_nr'] = aisdata['imo_nr'].astype('str')
        aisdata = aisdata[(aisdata['ShipData_ShiptypeLevel3'].notnull()) & (aisdata.groupby(
            'imo_nr').imo_nr.transform(len) > 5)]  # 9.12.19 change len from 1 to 10.
        aisdata = aisdata[['imo_nr', 'name', 'date_time_utc',
                           'lon', 'lat', 'ShipData_ShiptypeLevel3']]
        aisdata['date_time_utc'] = pd.to_datetime(aisdata.date_time_utc)
        # puts date time utc as index
        aisdata.set_index('date_time_utc', inplace=True)
        # only fishing and passenger vessels of interest
        vessels = aisdata[(aisdata.ShipData_ShiptypeLevel3 == 'Fish Catching') | (
            aisdata.ShipData_ShiptypeLevel3 == 'Passenger')]
        fish0720 = aisdata[(
            aisdata.ShipData_ShiptypeLevel3 == 'Fish Catching')]
        cruise0720 = aisdata[(aisdata.ShipData_ShiptypeLevel3 == 'Passenger')]
        imo = vessels.imo_nr.unique()
        numfish0720 = fish0720.imo_nr.unique()
        numcruise0720 = cruise0720.imo_nr.unique()
        allv = aisdata.imo_nr.unique()
        print('all', len(allv), 'fish', len(numfish0720),
              'cruise', len(numcruise0720))
        # print(imo, len(imo))
        #imo = aisdata[(aisdata.imo_nr == '8509181') | (aisdata.imo_nr == '9053282')] #
        # add vessel positions on map with minimum frequency of n minutes
        for only in imo:
            one = vessels.loc[vessels.imo_nr == only, :]
            # this adds a row every n minutes if there's no data then NAN is introduced.
            aisdata = one.resample(pd.offsets.Minute(n=60)).agg(
                {'imo_nr': 'last', 'name': 'last', 'lon': 'mean', 'lat': 'mean', 'ShipData_ShiptypeLevel3': 'last'})
            Fish = aisdata[aisdata.ShipData_ShiptypeLevel3 == 'Fish Catching']
            lat_fish = Fish.lat.tolist()
            lon_fish = Fish.lon.tolist()
            xf, yf = m(lon_fish, lat_fish)
            m.plot(xf, yf, 'go', markersize=2.5)
            Cruise = aisdata[aisdata.ShipData_ShiptypeLevel3 == 'Passenger']
            lat_cruise = Cruise.lat.tolist()
            lon_cruise = Cruise.lon.tolist()
            xc, yc = m(lon_cruise, lat_cruise)
            m.plot(xc, yc, 'yo', markersize=2.5)
        m.drawmapscale(4.8, 73.2, 8.8, 73.5, 120, units='km', fontsize=18, yoffset=None, barstyle='fancy',
                       labelstyle='simple', fillcolor1='grey', fillcolor2='white', fontcolor='grey', zorder=5)
        cbar = m.colorbar(ticks=(0, 15, 30, 45, 60, 75, 90, 100))
        cbar.ax.tick_params(labelsize=19)
        cbar.set_label(label='Ice concentration - [%]', fontsize=20)
        textstr = '{0} 2017'.format(t)
        props = dict(boxstyle='round', facecolor='w',
                     edgecolor='grey')  # 30.10 removed alpha = 0.7
        ax1.text(0.03, 0.97, textstr, transform=ax1.transAxes,
                 fontsize=32, verticalalignment='top', bbox=props)
        lab = mpatches.Patch(color='g', label='Fishing vessels')
        lab2 = mpatches.Patch(color='y', label='Passenger vessels')
        plt.legend(handles=[lab, lab2],  loc=4, fontsize=20)
        plt.tight_layout()
        plt.savefig(
            '/Users/ali/Desktop/Maps2020/grey_dpi300_20170{0}.png'.format(BLA))
        plt.show()


freqmaps()

# %%
# -------------------------------------------------------------------------------------------------------------
# code to make multipanel figure 3 and figure 3 continued


def multimaps():
    plt.figure(figsize=(7, 10), dpi=300)
    gs1 = gridspec.GridSpec(3, 11)
    gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.
    ax1 = plt.subplot(gs1[0, :5])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_nocb_201302.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[0, 5:])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_201702.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[1, :5])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_nocb_201304.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[1, 5:])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_201704.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[2, :5])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_nocb_201307.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[2, 5:])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_201707.png'))
    plt.imshow(im)
    plt.tight_layout()
    plt.savefig('/Users/ali/Desktop/Figure3_dpi300.eps', format='eps', bbox_inches='tight',
                pad_inches=0.05)
    plt.show()


multimaps()


def multimapscon():
    plt.figure(figsize=(7, 6.5), dpi=300)
    gs1 = gridspec.GridSpec(2, 11)
    gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

    ax1 = plt.subplot(gs1[0, :5])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_nocb_201309.png'))
    plt.imshow(im)

    ax1 = plt.subplot(gs1[0, 5:])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_201709.png'))
    plt.imshow(im)

    ax1 = plt.subplot(gs1[1, :5])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_nocb_201311.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[1, 5:])
    plt.axis('off')
    ax1.set_aspect('equal')
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/Maps2020/grey_dpi300_201711.png'))
    plt.imshow(im)
    plt.savefig('/Users/ali/Desktop/Figure3_continued_dpi300.eps', format='eps', bbox_inches='tight',
                pad_inches=0.05)
    plt.show()


multimapscon()

# %%
