#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
starting from the database of the linee separate (to create it, we have to run create_database_FCD_linee_separate.py script)
it creates the db of the velocities for each line
'''

'''
http://zetcode.com/db/sqlitepythontutorial/
http://sqlitebrowser.org/
'''

import sqlite3 as lite
import sys
import os
import numpy as np
from datetime import datetime
import datetime as dt
import osmnx as ox
from collections import Counter
import pdb
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import glob
import pandas as pd
from geopy.distance import VincentyDistance   #windows
import pandas as pd
from sqlalchemy import create_engine

import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
print '#1 Backend:',plt.get_backend()

def find_histogram_class_from_unicode(data_unicode):
    valoriEfrequenze=Counter(data_unicode)#serve per la moda: genera un array con due colonne (il valore dell'array e la relativa frequenza) e n classi (righe)
    n=len(valoriEfrequenze.most_common())# mi dice il numero delle classi dell'istogramma
    valori=np.zeros(n, dtype='<U4')
    occorrenze=np.zeros(n, dtype=int )
    #val_e_occ = np.zeros((n,2), dtype= 'U, int')
    #print n
    for i in xrange(0, n):#così ordina le frequenze in senso crescente:bins must increase monotonically
        #print i
        #pdb.set_trace(valoriEfrequenze)
        occorrenze[i]=valoriEfrequenze.most_common()[n-1-i][1]#primo indice righe, secondo colonne:la prima colonna è il valore, la seconda è la frequenza
        valori[i]=valoriEfrequenze.most_common()[n-1-i][0]
        #pdb.set_trace()
    moda=valori[n-1]
    return valori, occorrenze, moda,n# numpy, per questioni di memoria, non riesce a gestire un array le cui colonne sono di tipo diverso

def find_histogram_class_from_int(data_int):
    valoriEfrequenze=Counter(data_int)#serve per la moda: genera un array con due colonne (il valore dell'array e la relativa frequenza) e n classi (righe)
    n=len(valoriEfrequenze.most_common())# mi dice il numero delle classi dell'istogramma
    valori=np.zeros(n)
    occorrenze=np.zeros(n)
    for i in xrange(0, n):#così ordina le frequenze in senso crescente:bins must increase monotonically
         occorrenze[i]=valoriEfrequenze.most_common()[n-1-i][1]#primo indice righe, secondo colonne:la prima colonna è il valore, la seconda è la frequenza
         valori[i]=valoriEfrequenze.most_common()[n-1-i][0]
    moda=valori[n-1]
    return (np.column_stack((valori, occorrenze))).astype(int), moda,n


start = dt.datetime.now()

path = os.path.dirname(os.path.abspath(__file__))+'//..//' #esco dalla sub directory codice

# If not already present, create the directory to store the downloaded data
results_directory = 'velocities_nuove'
if not os.path.exists(path+'\\databases\\'+results_directory):
    os.makedirs(path+'\\databases\\'+results_directory)

database = '30000000righe'#'500000righe'
databases_names = glob.glob(path+'databases//'+database+'//*.db')
#sys.exit()
''' 
    # organizzazione database
    0 - index
    1 - linea
    2 - turno
    3 - date
    4 - mezzo
    5 - lat
    6 - lon
    7 - weekday The day of the week with Monday=0, Sunday=6
'''

# per la linea 15 (index 12), linea 38 (index 50), linea 62(index 89) non funziona il grafico
#linea 11, index 5
#linea 12, index 7
#linea 13, index 8
#linea 39, index 51
print 'Tot linee:', len(databases_names)

# ciclo sulle linee
for index in xrange(7, 9):#len(databases_names)):# index 12 corrisponde a linea 15, problematica
    linea = databases_names[index][len(path+'databases//'+database+'//csv_database_formatted'):-3]
    print index, '-esima linea:', linea
    con = lite.connect(databases_names[index])
    with con:

        cur = con.cursor()
        cur.execute('PRAGMA table_info(fcd_table)')
        metadata = cur.fetchall()

        for d in metadata:
            print d[0], d[1], d[2]

        # Dictionary cursor: we can refer to the data by their column names
        con.row_factory = lite.Row

        cur = con.cursor() 
        cur.execute("SELECT * FROM fcd_table")

        rows = cur.fetchall()
        # il problema era che i dati dentro iso_linea non sono più numeri, ma stringhe, 
        # per cui la maschera con un numero non funziona più
        # il dtype è fondamentale per evitare questo problema
        iso_linea = np.asarray(rows, dtype = object)
        rows = None 
        con = None
        print 'righe di dati per la linea:', len (iso_linea)
    if len(iso_linea) != 0:
        mezzo_histo = find_histogram_class_from_int(iso_linea[:,4])
        val_mezzo = mezzo_histo[0][:,0]
        occ_mezzo = mezzo_histo[0][:,1]

        vel_stats_linea = [] # statistiche di velocità relative a TUTTI i mezzi della linea
        print len(val_mezzo), ' mezzi per linea ',linea
        # CICLO SU TUTTI I MEZZI DELLA LINEA
        for j in xrange(0,len(val_mezzo)):
            vel_stats = [] # statistiche di velocità relative al singolo mezzo della linea
            mezzo = val_mezzo[j]
            print j, '-esimo mezzo:' , mezzo

            mask = iso_linea[:,4] == mezzo
            print 'righe per il mezzo',mezzo, ':', np.sum(mask)
            iso_mezzo  = iso_linea[mask]

            iso_mezzo[:,3] = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f') for date in iso_mezzo[:,3]]# [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f').date() for date in iso_mezzo[:,3]]

            date_sorted_iso_mezzo = iso_mezzo[np.argsort(iso_mezzo[:, 3])]
            
            # ciclo per calcolare la velocità (planimetrica)
            if len(date_sorted_iso_mezzo)>1:#
                for index_time in xrange(1,len(date_sorted_iso_mezzo)):
                    delta_t   = date_sorted_iso_mezzo[index_time,3] - date_sorted_iso_mezzo[index_time-1,3]
                    #print date_sorted_iso_mezzo[index_time,3]
                    #print date_sorted_iso_mezzo[index_time-1,3]
                    #print delta_t, delta_t.total_seconds()
                    delta_lon = date_sorted_iso_mezzo[index_time,6] - date_sorted_iso_mezzo[index_time-1,6]
                    delta_lat = date_sorted_iso_mezzo[index_time,5] - date_sorted_iso_mezzo[index_time-1,5]
                    # distanza in gradi 
                    spost_plan_degrees = np.sqrt(delta_lon*delta_lon+delta_lat*delta_lat)# degrees
                    # distanza in metri a partire da lon e lat
                    spost_plan_metres = np.float64(VincentyDistance( (date_sorted_iso_mezzo[index_time-1,5], date_sorted_iso_mezzo[index_time-1,6]) , 
                                                 (date_sorted_iso_mezzo[index_time,  5], date_sorted_iso_mezzo[index_time,6]) ).meters) # casto a np.float, così quando divido per zero, ottengo inf e non un errore
                    
                    if (delta_t.total_seconds() != 0) and (spost_plan_metres != 0):
                        vel = spost_plan_metres / delta_t.total_seconds() # meters / second
                        tempo_medio = date_sorted_iso_mezzo[index_time-1,3]+delta_t/2
                        lon_media = (date_sorted_iso_mezzo[index_time,6] + date_sorted_iso_mezzo[index_time-1,6])/2.0
                        lat_media = (date_sorted_iso_mezzo[index_time,5] + date_sorted_iso_mezzo[index_time-1,5])/2.0
                        start_time = date_sorted_iso_mezzo[index_time -1,3]
                        #0: mezzo, 1: tempo_medio, 2:lat_media, 3: lon_media, 4: velocità, 5: delta lat (dy), 6:delta_lon, 7: start lat (start y), 8: start lon (startx), 9: spost_plan-metri, 10:spost_plan_degrees, 11: dt, 12:start_time:
                        vel_stats_linea.append(np.array([mezzo, tempo_medio, lat_media, lon_media, vel, delta_lat, delta_lon, date_sorted_iso_mezzo[index_time-1,5], date_sorted_iso_mezzo[index_time-1,6], spost_plan_metres, spost_plan_degrees, delta_t.total_seconds(), start_time]))
            #else:
            #    vel_stats_linea.append(np.zeros(13)*np.nan)# la linea ha una sola posizione e non posso calcolare le velocità

        # Creo il vettore contenente le informazioni di tutti i mezzi della linea a partire dalla lista globale
        # lo salvo sottoforma di database
        array_vel_stats = np.array(vel_stats_linea)
        df = pd.DataFrame(array_vel_stats)
        df.columns = ['mezzo', 'tempo_medio', 'lat_media', 'lon_media', 'vel_m_s', 'delta_lat', 'delta_lon', 'start_lat' , 'start lon', 'spost_plan_metri', 'spost_plan_degrees', 'delta_t_seconds', 'start_time']
        csv_database_linea = create_engine('sqlite:///..///databases///'+results_directory+'///csv_database_velocities_'+linea+'.db')
        #The day of the week with Monday=0, Sunday=6
        # ultima colonna, per ora la numero 13: giorno della settimana
        df['weekday'] = df['start_time'].dt.dayofweek
        df.to_sql('fcd_table', csv_database_linea, if_exists='append', index = False)

        print '   vel min', np.min(array_vel_stats[:,4].astype(np.float64)), 'm/s , ', np.min(array_vel_stats[:,4].astype(np.float64)) * 3600 / 1000.0, 'km/h'
        print '   vel max', np.max(array_vel_stats[:,4].astype(np.float64)), 'm/s , ', np.max(array_vel_stats[:,4].astype(np.float64)) * 3600 / 1000.0, 'km/h'
        print ' vel media', np.nanmean(array_vel_stats[:,4].astype(np.float64)), 'm/s , ', np.mean(array_vel_stats[:,4].astype(np.float64)) * 3600 / 1000.0, 'km/h'
        print 'vel median', np.median(array_vel_stats[:,4].astype(np.float64)), 'm/s , ', np.median(array_vel_stats[:,4].astype(np.float64)) * 3600 / 1000.0, 'km/h'

        print '   min dt',  np.nanmin(array_vel_stats[:,11].astype(np.float64)), 'seconds'
        print '   max dt',  np.nanmax(array_vel_stats[:,11].astype(np.float64)), 'seconds'
        print '  media dt',  np.nanmean(array_vel_stats[:,11].astype(np.float64)), 'seconds'
        print ' median dt',  np.nanmedian(array_vel_stats[:,11].astype(np.float64)), 'seconds'
        print '    std dt', np.nanstd(array_vel_stats[:,11].astype(np.float64)), 'seconds'
        
        iso_linea = None
        array_vel_stats = None
        
print 'Database read in', (dt.datetime.now() - start).seconds, 'seconds'

plt.show()

