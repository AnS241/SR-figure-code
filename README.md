# SR-figure-code
# Python 3.7 - code for maps and figures in manuscript "Sea ice variability and maritime activity around Svalbard in the period 2012-2019"
# figure 1: lines XYZ - XYZ
# figure 2: lines XYZ - XYZ
# figure 3: lines XYZ - XYZ
# figure 4: lines XYZ - XYZ
def barlast():
    filenames = sorted(glob.glob('201*.csv'))
    print(filenames, len(filenames))
    barfish = []
    barcruise = []
    other = []
    for f in filenames:
        aisdata = pd.read_csv(f, delimiter=',', dtype={
            'imo_nr': str, 'length': str})
        aisdata['imo_nr'] = aisdata['imo_nr'].astype('str')
        aisdata = aisdata[(aisdata['ShipData_ShiptypeLevel3'].notnull()) & (aisdata.groupby(
            'imo_nr').imo_nr.transform(len) > 10) & (aisdata.lat > 72.2) & (aisdata.lon > -3.24)]
        Fish = aisdata[aisdata.ShipData_ShiptypeLevel3 == 'Fish Catching']
        # print(Fish.head(), Fish.shape)

        wanted = Fish[(Fish.lon > 10.3) & (
            Fish.lat >= 75.0)].imo_nr.unique()
        print(len(wanted))
        onlyfish = aisdata.loc[aisdata.ShipData_ShiptypeLevel3 ==
                               'Fish Catching', :].copy()
        typef = onlyfish.imo_nr.unique()
        # print('201804 fish: ', typef)
        barfish.append(len(typef))
        onlycruise = aisdata.loc[aisdata.ShipData_ShiptypeLevel3 ==
                                 'Passenger', :].copy()
        typec = onlycruise.imo_nr.unique()
        # print('201810cruise: ', typec)
        barcruise.append(len(typec))
        others = aisdata.loc[(aisdata['ShipData_ShiptypeLevel3'] != 'Fish Catching') & (
            aisdata['ShipData_ShiptypeLevel3'] != 'Passenger'), :]
        othertype = others.imo_nr.unique()
        # print('20104 other: ', othertype)
        other.append(len(othertype))
    print('number of fishing vessels:', barfish, len(barfish),
          'number of passenger vessels:', barcruise, len(barcruise),
          'all other vessels:', other, len(other))
    bars = np.add(barcruise, barfish).tolist()
    # , 2, 3, 4, 5, 6, 7, 8, ]#9, 10, 11,
    #r = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ]
    r = [0, 1, 2, 3, 4, 5, 6, 7]
    # 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # , 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',]# 'Oct',
    # names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    #          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    names = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']
    # 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    # r = list(range(0, 93))
    # names = [ '', '', '', '', '', '2018', '', '', '', '', '', '']#, '', '', '', '', '', '2019', '', '', '']
    barWidth = 1
    fig, ax1 = plt.subplots(figsize=(24, 9))  # , dpi = 300)
    plt.bar(r, barfish, color='g', edgecolor='white',
            width=barWidth, label='Fishing vessels')
    plt.bar(r, barcruise, bottom=barfish, color='y', edgecolor='white',
            width=barWidth, label='Passenger vessels')
    plt.bar(r, other, bottom=bars, color='grey', edgecolor='white',
            width=barWidth, label='Other vessels')
    # plt.ylim(top=250)
    plt.xticks(r, names, fontsize=20)  # ,fontweight = 'bold')
    plt.yticks(fontsize=20)
    plt.ylabel('number of vessel', fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 1, 0]
    plt.legend([handles[idx] for idx in order], [labels[idx]
                                                 for idx in order], loc='upper left', fontsize=34)
    # xt = [0,1]
    ax1.set_xticks([0], minor=True)  # ,12,24,36,48,60,72]
    # ax1.set_xticklabels(xt, fontsize =14)
    #ax1.xaxis.grid(True, which='both', linestyle='dotted', color='r')
    # ax1.grid(True, which='minor', axis='x', linestyle='dotted', color='pink')
    plt.tight_layout()
    plt.savefig(
        '/Users/ali/Desktop/barchartyears2508.png')
    plt.show()


barlast()
