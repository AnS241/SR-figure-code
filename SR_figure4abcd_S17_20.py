# %%
# code for timeseries figure 4 abcd and for Supplementary figures S17 to S20

import math
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.cbook import get_sample_data
import matplotlib.gridspec as gridspec
from PIL import Image
import glob
import datetime
from datetime import datetime, timedelta, date
import statistics
from statistics import stdev
from pandas import Series


# delimitations coordinates of the different areas

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

# first claculate sea ice cover and number of vessels per category for each area

os.chdir('/Volumes/AS_data/30012020/IceData/alldays')

with open('lonM.txt', 'r') as lo:
    dlon = lo.readlines()
LonIce = []
for line in dlon:
    item = line.split(',')
    for no in item:
        LonIce.append(float(no))


with open('latM.txt', 'r') as la:
    dlat = la.readlines()
LatIce = []
for line in dlat:
    item = line.split(',')
    for no in item:
        LatIce.append(float(no))


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
    df.to_csv(r'/Users/ali/Desktop/zoneallvessels2204.csv', index=True)
    dff.to_csv(r'/Users/ali/Desktop/zoneallfish2204.csv', index=True)
    dfc.to_csv(r'/Users/ali/Desktop/zoneallpass2204.csv', index=True)


all_zones_ships()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# second make the time series
# figure 4a) - 4d)


os.chdir('/Users/ali/Desktop/timeseries')


def monthlist():
    dates = ['2012-08-01', '2019-09-01']
    start, end = [datetime.strptime(_, '%Y-%m-%d') for _ in dates]
    def total_months(dt): return dt.month + 12 * dt.year
    mlist = []
    for tot_m in range(total_months(start)-1, total_months(end)):
        y, m = divmod(tot_m, 12)
        mlist.append(datetime(y, m+1, 1).strftime('%Y-%m'))
    return mlist


monthlist()


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
# fishing vessels
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
# passenger vessels
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


# -------------------------------------------------------------------------------------------------------------
# code to make multipanel figure 4


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
# %%
