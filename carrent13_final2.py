import numpy as np
import matplotlib.pyplot as plt

rentdata1 = np.array([['990101','1111','1030','1130','10','12','1030','1130','40'],
                     ['990101','1111','1100','1330','10','12','1100','1330','120'],
                     ['990329','1101','0830','1000','10','12','-','-','-'],
                     ['990329','1777','0915','1145','10','12','0920','1210','220'],
                     ['990520','1111','1000','1215','10','12','1111','1000','45'],
                     ['990620','1111','1005','1200','77','77','1005','1200','48'],
                     ['990607','1725','0915','1145','16','77','-','-','-'],
                    ['990720','1101','1000','1215','10','12','1111','1000','45'],
                     ['990820','1101','1005','1200','77','77','1005','1200','48']])

# convert main matrix  to integer
rentdata2= np.where(rentdata1 == '-',0,rentdata1)
rentdata2[rentdata2 == ''] = 0
rentdata2 = rentdata2.astype(np.int)

def time_interval(t1, t2):

    if t1<=t2:

        m1 = abs(t1) % 100
        m2 = abs(t2) % 100
        h1 = t1 // 100
        h2 = t2 // 100
        interval = ((h2-h1)*60)-m1+m2
    else :
        m1 = abs(t1) % 100
        m2 = abs(t2) % 100
        h1 = t1 // 100
        h2 = t2 // 100
        int = ((h2-h1)*60)-m1+m2
        interval = 1440 + int
    return interval
def days_statics(data,start, finish):

    demandlist = []
    cancellist = []
    distancelist = []
    average_list = []
    max = []
    min= []
    dvlist =[]

    findex = np.where(data[::,0] <= finish)[0][-1]+1
    sindex = np.where(data[::,0] >= start)[0][0]

    maindata = data[sindex:findex]
    if len(maindata) == 0:
        exit('no data for given period')
    dates = maindata[::, 0]
    dates = list(dict.fromkeys(dates))
    mainarr = np.zeros((len(dates), 0), dtype=int)


    for i in dates:
        g = np.where(maindata[::, 0] == i)
        maxdistance = np.amax(maindata[g, 8])
        mindistance = np.amin(maindata[g, 8])
        max.append(maxdistance)
        min.append(mindistance)
        cancels = np.count_nonzero(maindata[g, 8] == 0)
        demand = np.count_nonzero(maindata[g,0] == i)
        total_distance = maindata[g, 8].sum()
        average_distance = maindata[g, 8].mean()
        dv = maindata[g, 8].std()
        dvlist.append(dv)
        average_list.append(average_distance)
        distancelist.append(total_distance)
        cancellist.append(cancels)
        demandlist.append(demand)

    mainarr = np.insert(mainarr, 0, dates, axis=1)
    mainarr = np.insert(mainarr, 1, demandlist, axis=1)
    mainarr = np.insert(mainarr, 2, cancellist, axis=1)
    mainarr = np.insert(mainarr, 3, distancelist, axis=1)
    mainarr = np.insert(mainarr, 4, average_list, axis=1)
    mainarr = np.insert(mainarr, 5, dvlist, axis=1)
    mainarr = np.insert(mainarr, 6, max, axis=1)
    mainarr = np.insert(mainarr, 7, min, axis=1)

    return mainarr

def customer_matrix(data, start, finish):

    demandlist = []
    cancellist = []
    distancelist = []
    dvlist=[]
    findex = np.where(data[::,0] <= finish)[0][-1]+1
    sindex = np.where(data[::,0] >= start)[0][0]

    maindata = data[sindex:findex]

    customers = maindata[::, 1]
    customers = list(dict.fromkeys(customers))
    mainarr = np.zeros((len(customers), 0), dtype=int)

    for i in customers:
        cmat = np.where(maindata[::, 1] == i)
        datamatrix = maindata[cmat]
        cancels = np.count_nonzero(datamatrix[ ::,8] == 0)
        demand = np.count_nonzero(datamatrix[ ::,1] == i)
        total_distance = datamatrix[ ::,8].sum()
        var = datamatrix[ ::,8].std()
        dvlist.append(var)
        distancelist.append(total_distance)
        cancellist.append(cancels)
        demandlist.append(demand)

    mainarr = np.insert(mainarr, 0, customers, axis=1)
    mainarr = np.insert(mainarr, 1, demandlist, axis=1)
    mainarr = np.insert(mainarr, 2, cancellist, axis=1)
    mainarr = np.insert(mainarr, 3, distancelist, axis=1)
    mainarr = np.insert(mainarr, 4, dvlist, axis=1)


    return mainarr

def station_stats(data, start, finish):

    orglist = []
    deslist = []
    trips = []

    findex = np.where(data[::,0] <= finish)[0][-1]+1
    sindex = np.where(data[::,0] >= start)[0][0]

    maindata = data[sindex:findex]

    stations1 = maindata[::, 4]
    stations1 = list(dict.fromkeys(stations1))
    stations2 = maindata[::, 5]
    stations2 = list(dict.fromkeys(stations2))
    allstations = stations1 + stations2
    allstations = list(dict.fromkeys(allstations))

    mainarr = np.zeros((len(allstations), 0), dtype=int)

    for i in allstations:
        g = np.where(maindata[::, 4] == i)
        b = np.where(maindata[::, 5] == i)
        org = np.count_nonzero(maindata[g, 4] == i)
        des = np.count_nonzero(maindata[b, 5] == i)

        for j in stations2:
            y = [np.count_nonzero(maindata[g, 5] == j), i, j]
            trips.append(y)

        orglist.append(org)
        deslist.append(des)

    mainarr = np.insert(mainarr, 0, allstations, axis=1)
    mainarr = np.insert(mainarr, 1, orglist, axis=1)
    mainarr = np.insert(mainarr, 2, deslist, axis=1)

    return mainarr

def demand_time(data, start , finish):

    findex = np.where(data[::, 0] <= finish)[0][-1] + 1
    sindex = np.where(data[::, 0] >= start)[0][0]
    maindata = data[sindex:findex]
    dates = maindata[::, 0]
    dates = list(dict.fromkeys(dates))

    for i in dates:
        clock = 0

        tlist = []
        while clock <= 2400:
            dayind = np.where(maindata[::, 0] == i)
            daymatrix = maindata[dayind]
            ind = np.where(daymatrix[::,2]<=clock)
            mat1 = daymatrix[ind]
            ind2 = np.where(mat1[::,2]>clock-100)
            d = len(daymatrix[ind2])
            tlist.append(d)



            clock+=100

        x = np.arange(0,2500,100)
        x2 = np.arange(0,25)
        y = tlist
        plt.xlabel('Time(24H)'), plt.ylabel('demands quantity'), plt.title('quantity of demands by time ')
        plt.plot(x,y)
        plt.xticks(x,x2 )
    plt.show()


def trip_count(data, start, finish, code1, code2):
    findex = np.where(data[::, 0] <= finish)[0][-1] + 1
    sindex = np.where(data[::, 0] >= start)[0][0]

    maindata = data[sindex:findex]
    raftlist = []
    bargashtlist = []
    rinterval = []
    binterval = []
    rspeed = []
    bspeed = []


    dates = maindata[::, 0]
    dates = list(dict.fromkeys(dates))

    for i in dates:
        dayind = np.where(maindata[::, 0] == i)
        daymatrix = maindata[dayind]
        s_ind = np.where(daymatrix[::, 4] == code1)
        s_ind2 = np.where(daymatrix[::, 4] == code2)
        station_matrix = daymatrix[s_ind]
        station_matrix2 = daymatrix[s_ind2]
        
        bargasht = np.count_nonzero(station_matrix2[::, 5] == code1)
        raft = np.count_nonzero(station_matrix[::, 5] == code2)
        bargashtlist.append(bargasht)
        raftlist.append(raft)
        r_ind = np.where(station_matrix[::, 5] == code2)
        rmat = station_matrix[r_ind]

        for i in range(0, len(rmat)):  # raft intervals
            t_interval = time_interval(rmat[i, 6], rmat[i, 7])
            rinterval.append(t_interval)
            speed = rmat[i, 8] / (t_interval / 60)
            rspeed.append(speed)

        r_ind = np.where(station_matrix2[::, 5] == code1)
        rmat = station_matrix2[r_ind]
        for i in range(0, len(rmat)):  # raft intervals
            t_interval = time_interval(rmat[i, 6], rmat[i, 7])
            binterval.append(t_interval)
            speed = rmat[i, 8] / (t_interval / 60)
            bspeed.append(speed)

    return raftlist, bargashtlist, rinterval, binterval, rspeed, bspeed


def alltrips(data, start, finish):


    findex = np.where(data[::, 0] <= finish)[0][-1] + 1
    sindex = np.where(data[::, 0] >= start)[0][0]
    maindata = data[sindex:findex]

    stations1 = maindata[::, 4]
    stations1 = list(dict.fromkeys(stations1))
    stations2 = maindata[::, 5]
    stations2 = list(dict.fromkeys(stations2))
    allstations = stations1 + stations2
    allstations = list(dict.fromkeys(allstations))

    trips_arr = np.zeros((0, len(allstations)), dtype=int)
    trips_arr = np.insert(trips_arr, 0, allstations, axis=0)

    z = 0
    for i in allstations: # khoroji ha
        triplist = []
        station_ind = np.where(maindata[::, 4] == i)
        station_matrix = maindata[station_ind]
        #print('\n',station_matrix)
        for j in allstations:
            trip = np.count_nonzero(station_matrix[::, 5] == j)
            triplist.append(trip)
        z+=1
        trips_arr = np.insert(trips_arr, z, triplist, axis=0)

    return trips_arr

def scatter_size(data ,start , finish):

    findex = np.where(data[::,0] <= finish)[0][-1]+1
    sindex = np.where(data[::,0] >= start)[0][0]

    maindata = data[sindex:findex]
    stations = maindata[::,4:6]
    sizelist = []
    for i in stations:
        org = i[0]
        des = i[1]
        ind = np.where(maindata[::,4] == org)
        main_arr1 = maindata[ind]
        ind2 = np.where(main_arr1[::,5] == des)
        main_arr2 = main_arr1[ind2]
        size = len(main_arr2)*100
        sizelist.append(size)
    return sizelist

def unreliable_customers(data):
    total_deviation = []
    indicator = int(input('enter indicator for an unreliable customer (minutes): '))

    customers = data[::, 1]
    customers = list(dict.fromkeys(customers))

    for i in customers:
        cmat = np.where(data[::, 1] == i)
        cs_matrix = data[cmat]
        devlist = []
        for j in range(0,len(cs_matrix)):

            deviation = time_interval(cs_matrix[j, 2], cs_matrix[j, 6])
            devlist.append(deviation)

        total_deviation.append(sum(devlist))

    unreliable_list = []
    for i in range(0,len(customers)):
        if total_deviation[i] >  indicator:
            unreliable_list.append(customers[i])

    return 'unreliable customers codes: ',unreliable_list


def monthly_report(data, sdate, fdate):
    distance_list = []
    demand_list = []
    cancel_list = []


    m1 = abs(sdate // 100) % 100
    m2 = abs(fdate // 100) % 100
    period = np.arange(m1, m2 + 1)

    s_ind = np.where(m1 <= (abs(data[::, 0] // 100) % 100))[0][0]
    f_ind = np.where((abs(data[::, 0] // 100) % 100) <= m2)[0][-1] + 1
    maindata = data[s_ind:f_ind]
    j=1
    for i in period:
        mat = np.where((abs(maindata[::, 0] // 100) % 100) == i)[0]
        mainmat = maindata[mat]
        distance = mainmat[::, 8]
        demand = len(mainmat)
        cancel = np.count_nonzero(mainmat[::, 8] == 0)
        demand_list.append(demand)
        cancel_list.append(cancel)
        distance_list.append(distance)
        x2 = [str(x) for x in mainmat[::,0]]
        x = np.arange(0,len(mainmat[::,0]))
        y = distance
        a = len(period)
        ax1 = plt.subplot(a,2,j)
        ax1.plot(x,y)
        plt.xticks(x, x2)
        j+=1

        ax2 = plt.subplot(a,2,j)
        x=str(i)
        y=demand
        ax2.bar(x,y,label='demands')
        plt.legend()
        plt.minorticks_off()
        x = 1
        y=cancel
        ax2.bar(x,y,label='cancels')
        plt.legend()

        plt.minorticks_off()
        j+=1

    plt.show()
    return distance_list, demand_list, cancel_list, period

def cs_scatter(data,sdate, fdate):
    lsx =[]
    lsy= []
    sizelist2 = []


    cs_code = int(input('enter customer code: '))
    m1 = abs(sdate // 100) % 100
    m2 = abs(fdate // 100) % 100
    periods = np.arange(m1, m2 + 1)

    s_ind = np.where(m1 <= (abs(data[::, 0] // 100) % 100))[0][0]
    f_ind = np.where((abs(data[::, 0] // 100) % 100) <= m2)[0][-1] + 1
    maindata = data[s_ind:f_ind]
    a = len(periods)
    for i in periods:

        mat = np.where((abs(maindata[::, 0] // 100) % 100) == i)[0]
        mainmat = maindata[mat]

        c_ind = np.where(mainmat[::,1] == cs_code)
        c_arr = mainmat[c_ind]

        x1 = c_arr[::, 4]
        x2 = [str(x) for x in x1]
        y = c_arr[::, 5]
        y2 = [str(x) for x in y]

        lsx.append(x2)
        lsy.append(y2)

        stations = c_arr[::, 4:6]
        sizelist = []
        for i in stations:

            org = i[0]
            des = i[1]
            ind = np.where(mainmat[::, 4] == org)
            main_arr1 = mainmat[ind]
            ind2 = np.where(main_arr1[::, 5] == des)
            main_arr2 = main_arr1[ind2]
            size = len(main_arr2) * 100
            sizelist.append(size)
        sizelist2.append(sizelist)

    j=1
    for i in range(0, len(periods)):
        x = lsx[i]
        y = lsy[i]
        sz = sizelist2[i]

        ax = plt.subplot(len(periods),1,j)
        ax.scatter(x,y,s=sz)
        j+=1
    plt.show()



def main():
    print('*'*20,'reports','*'*27)

    startdate = int(input('enter start date: '))
    finishdate = int(input('enter finish date: '))

    arr1 = days_statics(rentdata2, startdate, finishdate)


    # statistical 1
    print('total demands : ', arr1[::,1].sum())
    print('average demands per day : ', arr1[::, 1].mean())
    print('demands standard deviation : ', arr1[::, 1].std())
    print('total cancels : ', arr1[::, 2].sum())
    print('total distance : ', arr1[::, 3].sum())
    print('average distance per day : ', arr1[::, 3].mean())
    print('distance standard deviation : ', arr1[::, 3].std())
    print('*' * 20, 'end of reports', '*' * 20)

    ######################################################
    # plot 1

    x1 = customer_matrix(rentdata2, startdate, finishdate)[::,0]
    x2 = [str(x) for x in x1]
    y = customer_matrix(rentdata2, startdate, finishdate)[::,3]/customer_matrix(rentdata2, startdate, finishdate)[::,1]
    std = customer_matrix(rentdata2, startdate, finishdate)[::, 4]
    plt.bar(x2, y, yerr=std,color='g',error_kw={'ecolor':'0.01','capsize':5})
    plt.title('average distance by customer')
    plt.xlabel('customer code')
    plt.ylabel('average distance')

    plt.show()

    #############################################
    date_len = len(days_statics(rentdata2,startdate,finishdate))
    x = np.arange(0,date_len)
    x1 = days_statics(rentdata2,startdate,finishdate)[::,0]
    x2 = [str(x) for x in x1]  # dates
    y1 = days_statics(rentdata2,startdate,finishdate)[::,4]  # average
    y2 = days_statics(rentdata2,startdate,finishdate)[::,6]  # max
    y3 = days_statics(rentdata2,startdate,finishdate)[::,7]  # min
    err = days_statics(rentdata2,startdate,finishdate)[::,5]
    w = 0.25
    plt.bar(x, y1,w,yerr=err,error_kw={'ecolor':'0.01','capsize':3},label='averge distance')
    plt.bar(x+w, y2,w,label='max distance')
    plt.bar(x + w*2, y3, w, label='min distance')
    plt.legend()
    plt.xlabel('date')
    plt.ylabel('distance')
    plt.title('distance statistics per day')
    plt.xticks(x+w,x2)
    plt.show()
    # plot2 ************************************
    names = arr1[::,0]
    xv = np.arange(0, len(arr1))
    w=0.35
    value1 = arr1[::,1]-arr1[::,2]
    value2 = arr1[::,2]
    plt.bar(xv, value1,w, label = 'trips')
    plt.bar(xv, value2,w, label = 'cancels',bottom=value1)
    plt.xticks(xv, names)
    plt.legend()
    plt.xlabel('dates')
    plt.title('number of trips and cancels by dates')
    plt.show()
    ###########################################
    stationmatrix = station_stats(rentdata2,startdate,finishdate)
    names = stationmatrix[::, 0]
    xv = np.arange(0,len(stationmatrix))
    value1 = stationmatrix[::, 1]
    value2 = stationmatrix[::, 2]
    w=0.35
    plt.bar(xv, value1, w, label='khoroj')
    plt.bar(xv+w, value2, w, label='vorod')
    plt.xticks(xv+w/2,names)
    plt.xlabel('stations')
    plt.legend()
    plt.title('vorod va khoroj')
    plt.show()
    #########################################################
    # demand by time plot
    demand_time(rentdata2,startdate,finishdate)


    # scatter 1 ***************************************************
    findex = np.where(rentdata2[::, 0] <= finishdate)[0][-1] + 1
    sindex = np.where(rentdata2[::, 0] >= startdate)[0][0]

    maindata = rentdata2[sindex:findex]

    plt.scatter(maindata[::, 4], maindata[::, 5], s=scatter_size(rentdata2, startdate, finishdate))
    plt.xticks(maindata[::, 4], maindata[::, 4])
    plt.yticks(maindata[::, 5], maindata[::, 5])
    plt.title('From To scatter')
    plt.xlabel('destination')
    plt.ylabel('origin')
    plt.show()
    #  ************************
    code01 = int(input('enter station 1: '))
    code02 = int(input('enter station 2: '))
    print('*' * 20, 'reports 2', '*' * 27)
    print('total demands(raft): ',sum(trip_count(rentdata2, startdate, finishdate,code01,code02)[0]))
    print('average per day(raft) : ', np.mean(trip_count(rentdata2, startdate, finishdate, code01, code02)[0]))
    print('deviation(raft) : ', np.std(trip_count(rentdata2, startdate, finishdate, code01, code02)[1]))
    print('total demands(bargasht) : ', sum(trip_count(rentdata2, startdate, finishdate, code01, code02)[1]))
    print('average per day(bargasht) : ', np.mean(trip_count(rentdata2, startdate, finishdate, code01, code02)[1]))
    print('deviation (bargasht) : ', np.std(trip_count(rentdata2, startdate, finishdate, code01, code02)[1]))
    if len(trip_count(rentdata2, startdate, finishdate, code01, code02)[2]) > 0:
        print('average duration : ', np.mean(trip_count(rentdata2, startdate, finishdate, code01, code02)[2]))
        print(' duration SD : ', np.std(trip_count(rentdata2, startdate, finishdate, code01, code02)[2]))
        print('MAX duration : ', max(trip_count(rentdata2, startdate, finishdate, code01, code02)[2]))
        print('MIN duration : ', min(trip_count(rentdata2, startdate, finishdate, code01, code02)[2]))
        print('average speed (raft): ', np.mean(trip_count(rentdata2, startdate, finishdate, code01, code02)[4]))
    if len(trip_count(rentdata2, startdate, finishdate, code01, code02)[3]) > 0:
        print('average duration (b) : ', np.mean(trip_count(rentdata2, startdate, finishdate, code01, code02)[3]))
        print(' duration SD (b) : ', np.std(trip_count(rentdata2, startdate, finishdate, code01, code02)[3]))
        print('MAX duration (b): ', max(trip_count(rentdata2, startdate, finishdate, code01, code02)[3]))
        print('MIN duration (b) : ', min(trip_count(rentdata2, startdate, finishdate, code01, code02)[3]))
        print('average speed (b): ', np.mean(trip_count(rentdata2, startdate, finishdate, code01, code02)[5]))

    ################### pie charts
    tripsarr = alltrips(rentdata2, startdate, finishdate)
    labels = alltrips(rentdata2, startdate, finishdate)[0,::]
    ind1 = np.where(labels == code01)[0]
    sizes = tripsarr[ind1+1][0]
    ax1 = plt.subplot(221)
    ax1.pie(sizes,labels=labels,autopct='%1.1f%%')
    ax1.set_title('from code 1')
    sizes2 = list(tripsarr[1::,ind1].T)[0]
    ax2 = plt.subplot(222)
    ax2.set_title('to code 1')
    ax2.pie(sizes2, labels=labels, autopct='%1.1f%%')
    ax3 = plt.subplot(223)
    ind1 = np.where(labels == code02)[0]
    sizes = tripsarr[ind1+1][0]
    ax3.set_title('from code 2')
    ax3.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax4 = plt.subplot(224)
    ind1 = np.where(labels == code02)[0]
    sizes = list(tripsarr[1::,ind1].T)[0]
    ax4.set_title('to code 2')
    ax4.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.show()
    ###########################################################
    print('*' * 20, 'end of reports', '*' * 20)

    # *************************monthly charts*******************************
    print('***** monthly reports *****')
    stdate = int(input('enter start date : '))
    fidate = int(input('enter finish date: '))
    monthly_report(rentdata2,stdate,fidate)
    cs_scatter(rentdata2, stdate, fidate)

    #####

    print(unreliable_customers(rentdata2))

main()
