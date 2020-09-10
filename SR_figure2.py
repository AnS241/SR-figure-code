
# %%
# code for the creation of figure 2a) and 2b)

import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.cbook import get_sample_data
import matplotlib.gridspec as gridspec
from numpy.random import rand
from PIL import Image
import glob


# figure 2a) - barchart monthly number of vessels per category

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
    # to add space use \n
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

# -------------------------------------------------------------------------------------------------------------
# figure 2b) - number of fihing vessels per month

df = pd.read_csv("fishingvessels.csv", delimiter=',')
# colourblind friendly - brown & blue (dark-light-dark)
colors = ['#8c510a', '#bf812d', '#dfc27d', '#e8dab7',
          '#c7eae5', '#80cdc1', '#35978f', '#01665e', 'k']

markers = ['o', '^', 's', 'd', 'D', 'P', 'X', 'p', '']
# multiple line plot
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
# Add legend
plt.legend(loc=2, ncol=2, fontsize=28, facecolor='white')
plt.grid(True, linestyle='dotted')
plt.tight_layout()
plt.savefig('/Users/ali/Desktop/spagplot_fish_bb_0207.png')

plt.show()


# -------------------------------------------------------------------------------------------------------------
# code to make multipanel figure 2


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


multifig2()

# %%
