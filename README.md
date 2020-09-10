# SR-figure-code
# Python 3.7 - code for maps and figures in manuscript "Sea ice variability and maritime activity around Svalbard in the period 2012-2019"
# figure 1: lines 58 - 218
# figure 2: lines 219 - 351
# figure 3 and S1 - S15: lines 352 - 538
# figure 4a) - 4d): lines 540 - 811
# figure 4e) and S16: lines 812 - 1034
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
    
------------------------------------------------------------------------------------------------------------------------------------------------------------------

#figure 1a) - map of Svalbard with settlements 

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

#figure 1b) - map of geographic location in the Arctic

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

------------------------------------------------------------------------------------------------------------------------------------------------------------------

#figure 2a) - barchart monthly number of vessels per category

def barplots(dir):
    os.chdir(dir)
    filenames = sorted(glob.glob('ais_ST*.csv'))
    print(filenames, len(filenames))
    barfish = []
    barcruise = []
    other = []
    for f in filenames:
        aisdata = pd.read_csv(f, delimiter=',', dtype={
            'imo_nr': str, 'length': str})
        aisdata['imo_nr'] = aisdata['imo_nr'].astype('str')
        aisdata = aisdata[(aisdata['ShipData_ShiptypeLevel3'].notnull()) & (aisdata.groupby(
            'imo_nr').imo_nr.transform(len) > 10)]
        onlyfish = aisdata.loc[aisdata.ShipData_ShiptypeLevel3 ==
                               'Fish Catching', :].copy()
        typef = onlyfish.imo_nr.unique()
        barfish.append(len(typef))
        onlycruise = aisdata.loc[aisdata.ShipData_ShiptypeLevel3 ==
                                 'Passenger', :].copy()
        typec = onlycruise.imo_nr.unique()
        barcruise.append(len(typec))
        others = aisdata.loc[(aisdata['ShipData_ShiptypeLevel3'] != 'Fish Catching') & (
            aisdata['ShipData_ShiptypeLevel3'] != 'Passenger'), :]
        othertype = others.imo_nr.unique()
        other.append(len(othertype))
    print('number of fishing vessels:', barfish, len(barfish),
          'number of passenger vessels:', barcruise, len(barcruise),
          'all other vessels:', other, len(other))
    bars = np.add(barcruise, barfish).tolist()
    r = list(range(0, 86))
    #to add space use \n
    names = ['\n2012', 'Sep', '', '', '', '', '', 'Mar', '', '', '\n2013', '', '', 'Sep', '', '', '',
             '', '', 'Mar', '', '', '\n2014', '', '', 'Sep', '', '', '', '', '', 'Mar', '', '', '\n2015', '', '', 'Sep', '', '', '',
             '', '', 'Mar', '', '', '\n2016', '', '', 'Sep', '', '', '', '', '', 'Mar', '', '', '\n2017', '', '', 'Sep', '', '', '',
             '', '', 'Mar', '', '', '\n2018', '', '', 'Sep', '', '', '', '', '', 'Mar', '', '', '\n2019', '', '', 'Sep']
    barWidth = 1
    fig, ax1 = plt.subplots(figsize=(24, 9), dpi=300)
    plt.bar(r, barfish, color='g', edgecolor='white',
            width=barWidth, label='Fishing vessels')
    plt.bar(r, barcruise, bottom=barfish, color='y', edgecolor='white',
            width=barWidth, label='Passenger vessels')
    plt.bar(r, other, bottom=bars, color='grey', edgecolor='white',
            width=barWidth, label='Other vessels')
    plt.yticks(fontsize=22)
    plt.xticks(r, names, fontsize=22) 
    ax1.set_xticks([1, 7, 13, 23, 29, 35, 41, 47, 53,
                    59, 65, 71, 77, 83], minor=True)
    lines = [5, 17, 29, 41, 53, 65, 77]
    for l in lines:
        plt.axvline(l, linestyle='dotted', color='k')
    plt.ylim(top=220)
    plt.ylabel('number of vessels', fontsize=25)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx]
                                                 for idx in order], loc='upper left', fontsize=28)
    plt.tight_layout()
    plt.savefig('/Users/ali/Desktop/barchart_1908.png')
    plt.show()


barplots("/Users/ali/Desktop/ships/STcut")


#figure 2b) - number of fihing vessels per month

df = pd.read_csv("fishingvessels.csv", delimiter=',')
#colourblind friendly - brown & blue (dark-light-dark)
colors = ['#8c510a', '#bf812d', '#dfc27d', '#e8dab7',
          '#c7eae5', '#80cdc1', '#35978f', '#01665e', 'k']

markers = ['o', '^', 's', 'd', 'D', 'P', 'X', 'p', '']
#multiple line plot
plt.figure(figsize=(24, 9), dpi=300)
num = 0
for column in df.drop('x', axis=1):
    plt.plot(df['x'], df[column], marker=markers[num], color=colors[num],
             linewidth=2.5, markersize=8, label=column)
    num += 1
plt.yticks(fontsize=22)
plt.ylabel('number of fishing vessels', fontsize=25)
r = list(range(1, 13))
names = ['J', 'F', 'M', 'A', 'M', 'J',
         'J', 'A', 'S', 'O', 'N', 'D']

plt.xticks(r, names, fontsize=22)
#Add legend
plt.legend(loc=2, ncol=2, fontsize=28, facecolor='white')
plt.grid(True, linestyle='dotted')
plt.tight_layout()
plt.savefig('/Users/ali/Desktop/spagplot_fish_bb_0207.png')

plt.show()


def multifig2():
    plt.figure(figsize=(11, 8), dpi=300)
    gs1 = gridspec.GridSpec(2, 1)
    gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.
    ax1 = plt.subplot(gs1[0, :])
    plt.axis('on')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.text(0, 0.9, 'a)', fontsize=16, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax1.transAxes)
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/timeseries/barchart_1908.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[1, :])
    plt.axis('on')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.text(0, 0.9, 'b)', fontsize=16, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax1.transAxes)
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/timeseries/spagplot_fish_bb_0207.png'))
    plt.imshow(im)
    plt.savefig('/Users/ali/Desktop/Figure2_dpi300.eps', format='eps', bbox_inches='tight',
                pad_inches=0.05)
    plt.show()

#multifig2()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

#figure 3 and S1 - S15

os.chdir('/Volumes/AS_data/IceData')

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
        # building the map
        fig = plt.figure(figsize=(12, 10), dpi=300)
        # with cb figsize=(12,10)
        fig, ax1 = plt.subplots(figsize=(12, 10))  # , dpi=300)
        zoom_scale = 0
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
        plt.savefig('/Users/ali/Desktop/Maps2020/grey_dpi300_20170{0}.png'.format(BLA))
        plt.show()

freqmaps()


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

------------------------------------------------------------------------------------------------------------------------------------------------------------------

#figure 4a) - 4d)

#delimitations coordinates of the different areas

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

#first claculate sea ice cover and number per vessels for each area

os.chdir('/Volumes/AS_data/30012020/IceData/alldays')


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

x = 1

def all_zones_ice():
    month = []
    zoneheaders = ['z1', 'z2', 'z3', 'z4', 'z5',
                   'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']
    for counter, file in enumerate(glob.glob('IUP_ASI_20*')):
        with open(file, 'r') as f:
            data = f.readlines()
        seaice = []
        for line in data:
            item = line.split(',')
            for ni in item:
                seaice.append(float(ni))
        dataset = pd.DataFrame(
            {'lat': LatIce, 'lon': LonIce, 'seaice': seaice})
        # print(dataset)
        day = []
        for row in arearows:
            lw = float(row[1])
            le = float(row[2])
            ls = float(row[3])
            ln = float(row[4])
            # print(lw,le,ls,ln)
            zone = dataset.loc[(dataset['lon'] >= lw) & (dataset['lon'] <= le) & (
                dataset['lat'] >= ls) & (dataset['lat'] <= ln)]
            noland = zone.dropna()  # removes points that are on land
            ice_area = noland.loc[(noland['seaice'] >= 15.0)]
            TPA = len(noland)  # total number of points on water in zone
            TPI = len(ice_area)  # total number of points with sea ice
            PZCI = (TPI * 100)/TPA  # the percentage of sea coverage in zone
            # print('the percentage sea ice cover in area',row[0],'is: ', '%.3f' % (PZCI)) # %.3f % (X) limits the number after decimal point
            # make list of sea ice concentration for each zone each day
            day.append('%.3f' % (PZCI))
        # print(day)
        month.append(day)
    # print(month)
    df = pd.DataFrame(index=pd.date_range(
        '2012-08-01', '2019-09-30', freq='D'), data=month, columns=zoneheaders)
    # print (df)
    df.to_csv(r'/Users/ali/Desktop/all12zones_ice2204.csv', index=True)
    
all_zones_ice()

def all_zones_ships():
    years = []
    year_fish = []
    year_cruise = []
    zoneheaders = ['z1', 'z2', 'z3', 'z4', 'z5',
                   'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z12']
    # filenames = sorted(glob.glob('ais_ST*.csv'))
    for counter, file in enumerate(sorted(glob.glob('ais_ST20*'))):
        print(file)
        aisdata = pd.read_csv(file, delimiter=',', dtype={
            'imo_nr': str, 'length': str})
        aisdata['imo_nr'] = aisdata['imo_nr'].astype('str')
        aisdata = aisdata[(aisdata['ShipData_ShiptypeLevel3'].notnull()) & (
            aisdata.groupby('imo_nr').imo_nr.transform(len) > 5)]
        vessels = aisdata[(aisdata.ShipData_ShiptypeLevel3 == 'Fish Catching') | (
            aisdata.ShipData_ShiptypeLevel3 == 'Passenger')]
        Fish = aisdata[aisdata.ShipData_ShiptypeLevel3 == 'Fish Catching']
        Cruise = aisdata[aisdata.ShipData_ShiptypeLevel3 == 'Passenger']
        month = []
        month_fish = []
        month_cruise = []
        for row in arearows:
            lw = float(row[1])
            le = float(row[2])
            ls = float(row[3])
            ln = float(row[4])
            # print(lw,le,ls,ln)
            fish_and_cruise = vessels.loc[(vessels['lon'] >= lw) & (
                vessels['lon'] <= le) & (vessels['lat'] >= ls) & (vessels['lat'] <= ln)]
            numves = fish_and_cruise.imo_nr.unique()
            data = Fish.loc[(Fish['lon'] >= lw) & (Fish['lon'] <= le) & (
                Fish['lat'] >= ls) & (Fish['lat'] <= ln)]
            numfish = data.imo_nr.unique()
            datapass = Cruise.loc[(Cruise['lon'] >= lw) & (Cruise['lon'] <= le) & (
                Cruise['lat'] >= ls) & (Cruise['lat'] <= ln)]
            numpass = datapass.imo_nr.unique()
            # print('the number of vessel in area', row[0], 'is', len(numves),
            #      'the number of fishing vessels in area', row[0], 'is:', len(
            #    numfish),
            #    'the number of passenger vessels in area', row[0], 'is', len(numpass))
            # headers = ['areas', 'all vessels',
            #          'fishing vessels', 'passenger vessels']
            # areas = [row[0], len(numves), len(numfish), len(numpass)]
            month.append(len(numves))
            month_fish.append(len(numfish))
            month_cruise.append(len(numpass))
        # print(month)
        years.append(month)
        year_fish.append(month_fish)
        year_cruise.append(month_cruise)
    print(years)
    df = pd.DataFrame(index=pd.date_range(
        '2012-08-01', '2019-09-30', freq='M'), data=years, columns=zoneheaders)
    dff = pd.DataFrame(index=pd.date_range(
        '2012-08-01', '2019-09-30', freq='M'), data=year_fish, columns=zoneheaders)
    dfc = pd.DataFrame(index=pd.date_range(
        '2012-08-01', '2019-09-30', freq='M'), data=year_cruise, columns=zoneheaders)
    # df.to_csv(r'/Users/ali/Desktop/zoneallvessels2204.csv', index=True)
    # dff.to_csv(r'/Users/ali/Desktop/zoneallfish2204.csv', index=True)
    # dfc.to_csv(r'/Users/ali/Desktop/zoneallpass2204.csv', index=True)


all_zones_ships()

#second make the time series

os.chdir('/Users/ali/Desktop/timeseries')

def timeseries():
    allzonesIce = pd.read_csv('all12zones_ice2204.csv', delimiter=',')
    print(allzonesIce)
    allzonesIce['date'] = pd.to_datetime(
        allzonesIce['date'], format='%Y-%m-%d')
    allzonesfish = pd.read_csv('zoneallfish2204.csv', delimiter=',')
    allzonesfish['date'] = pd.to_datetime(
        allzonesfish['date'], format='%Y-%m-%d')
    allzonespass = pd.read_csv('zoneallpass2204.csv', delimiter=',')
    allzonespass['date'] = pd.to_datetime(
        allzonespass['date'], format='%Y-%m-%d')
    for n in range(1, 13):  # /!\ changed for presentation original -> for n in range(1,13)
        tsi = pd.Series(np.array(allzonesIce.iloc[:, n]), name=(
            'zone', n), index=allzonesIce['date'])
        tsf = pd.Series(np.array(allzonesfish.iloc[:, n]), name=(
            'zone', n), index=allzonesfish['date'])
        tsp = pd.Series(np.array(allzonespass.iloc[:, n]), name=(
            'zone', n), index=allzonespass['date'])
        allmonths = []
        allfish = []
        allpass = []
        listmeans = []
        lstdev = []
        # pd.date_range('2012-08','2018-08', freq = 'M'): # between two dates, frequency is months.
        for m in monthlist():
            month = tsi[str(m)]  # ex: august2012 = ts['2012-8']
            mf = tsf[str(m)]
            mp = tsp[str(m)]
            mm = month.mean()
            std = statistics.stdev(month)
            allmonths.append(m)
            listmeans.append(mm)
            lstdev.append(std)
            mmf = mf.mean()
            allfish.append(mmf)
            mmp = mp.mean()
            allpass.append(mmp)
            # print('montly mean of zone',n,'in', m ,'is:', mm, 'with a standard deviation of:', std)
            # form this data plot a timeseries
        time = np.array(allmonths)
        zone = np.array(listmeans)
        standev = np.array(lstdev)
        fish = np.array(allfish)
        cruise = np.array(allpass)
#fishing vessels
        # before figsize =(36, 12)
        fig, ax1 = plt.subplots(figsize=(36, 8), dpi=300)
        ax1.plot(time, zone, 'b-', label='Mean ice extent', linewidth=5)
        ax1.fill_between(range(86), zone-standev, zone +
                         standev, color='b', alpha=0.2)
        # ax1.set_xlabel('time (month)')
        ax1.set_ylabel('Ice extent - [%]', fontsize=42)
        ax1.set_ylim(bottom=0, top=100)
        ax2 = ax1.twinx()
        ax2.plot(time, fish, 'g-', label='Fishing vessels', linewidth=6)
        ax2.set_ylabel('Number of vessels', fontsize=42)
        ax2.set_ylim(bottom=0, top=120)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        textstr = 'Zone {0}'.format(n)
        props = dict(boxstyle='round', facecolor='w',
                     alpha=0.7, edgecolor='grey')
        ax1.text(0.9, 0.97, textstr, transform=ax1.transAxes,
                 fontsize=36, verticalalignment='top', bbox=props)
        # fig.tight_layout()
        names = ['\n2012', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2013', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2014', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2015', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2016', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2017', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2018', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2019', '', '', 'Sep']
        plt.xticks(time, names, fontweight='bold', fontsize=34)
        lines = [1, 7, 13, 19, 25, 31, 37, 43,
                 49, 55, 61, 67, 73, 79, 85]
        for l in lines:
            plt.axvline(l, linestyle='dotted', color='k')
        ax1.margins(x=0.01, y=0)
        ax1.tick_params(axis='x', labelsize=42)
        ax1.tick_params(axis='y', labelsize=42)
        ax2.tick_params(axis='y', labelsize=42)
        plt.legend(h1+h2, l1+l2, loc=2, fontsize=44)
        plt.tight_layout()
        plt.savefig(
            '/Users/ali/Desktop/timeseries/dpi300_0307_TSF{0}.png'.format(n))
        # plt.show()
#passenger vessels
        # before figsize =(36, 12)
        fig, ax1 = plt.subplots(figsize=(36, 8), dpi=300)
        ax1.plot(time, zone, 'b-', label='Mean ice extent', linewidth=5)
        ax1.fill_between(range(86), zone-standev, zone +
                         standev, color='b', alpha=0.2)
        # ax1.set_xlabel('time (month)')
        ax1.set_ylabel('Ice extent - [%]', fontsize=42)  # 32
        ax1.set_ylim(bottom=0, top=100)
        ax2 = ax1.twinx()
        ax2.plot(time, cruise, 'y-', label='Passenger vessels', linewidth=6)
        ax2.set_ylabel('Number of vessels', fontsize=42)
        ax2.set_ylim(bottom=0, top=35)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        textstr = 'Zone {0}'.format(n)
        props = dict(boxstyle='round', facecolor='w',
                     alpha=0.7, edgecolor='grey')
        ax1.text(0.88, 0.97, textstr, transform=ax1.transAxes,
                 fontsize=36, verticalalignment='top', bbox=props)
        # fig.tight_layout()
        names = ['\n2012', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2013', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2014', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2015', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2016', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2017', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2018', '', '', 'Sep', '', '', '',
                 '', '', 'Mar', '', '', '\n2019', '', '', 'Sep']
        plt.xticks(time, names, fontweight='bold')
        lines = [1, 7, 13, 19, 25, 31, 37, 43,
                 49, 55, 61, 67, 73, 79, 85]
        for l in lines:
            plt.axvline(l, linestyle='dotted', color='k')
        ax1.margins(x=0.01, y=0)
        ax1.tick_params(axis='x', labelsize=42)  # 30
        ax1.tick_params(axis='y', labelsize=42)
        ax2.tick_params(axis='y', labelsize=42)
        plt.legend(h1+h2, l1+l2, loc=2, fontsize=44)
        plt.tight_layout()
        plt.savefig(
            '/Users/ali/Desktop/timeseries/dpi300_0307_TSP{0}.png'.format(n))
        # plt.show()


timeseries()

------------------------------------------------------------------------------------------------------------------------------------------------------------------

#figure 4e) - map of Svalbard with delimitation of 12 zones

def areas_rec():
    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    zoom_scale = 0
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
    newrows = []
    for row in arearows:
        lw = float(row[1])
        le = float(row[2])
        ls = float(row[3])
        ln = float(row[4])
        # area('#5ab4ac',4) dark purple: #54278f , dark blue: #045a8d
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

def multiTS():
    plt.figure(figsize=(13, 14), dpi=300)
    gs1 = gridspec.GridSpec(5, 3)
    gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.
    ax1 = plt.subplot(gs1[0, :])
    plt.axis('on')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.text(0, 0.9, 'a)', fontsize=16, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax1.transAxes)
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/timeseries/dpi300_TSP2.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[1, :])
    plt.axis('on')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.text(0, 0.9, 'b)', fontsize=16, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax1.transAxes)
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/timeseries/dpi300_TSP8.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[2, :])
    plt.axis('on')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.text(0, 0.9, 'c)', fontsize=16, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax1.transAxes)
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/timeseries/dpi300_TSF6.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[3, :])
    plt.axis('on')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.text(0, 0.9, 'd)', fontsize=16, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax1.transAxes)
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/timeseries/dpi300_TSF10.png'))
    plt.imshow(im)
    ax1 = plt.subplot(gs1[-1, 0])
    plt.axis('on')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_aspect('equal')
    ax1.text(0, 0.9, 'e)', fontsize=16, fontweight='bold',
             horizontalalignment='right',
             verticalalignment='bottom',
             transform=ax1.transAxes)
    im = plt.imread(get_sample_data(
        '/Users/ali/Desktop/timeseries/areas_dpi300.png'))
    plt.imshow(im)
    plt.tight_layout()
    plt.savefig('/Users/ali/Desktop/Figure4_dpi300.eps', format='eps', bbox_inches='tight',
                pad_inches=0.05)
    plt.show()


multiTS()
