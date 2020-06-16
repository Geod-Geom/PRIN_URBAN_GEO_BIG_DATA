#!/usr/bin/python
# -*- coding: utf-8 -*-

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

def distance_on_unit_sphere(lat1, long1, lat2, long2):
# https://www.johndcook.com/blog/python_longitude_latitude/
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = math.pi/180.0
     
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
     
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
     
    # Compute spherical distance from spherical coordinates.
     
    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    # sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
     
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
    math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
     
    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc

start = dt.datetime.now()

path = os.path.dirname(os.path.abspath(__file__))+'//..//' #esco dalla sub directory codice

#name_database = path + '\\fcd\\csv_database_formatted_short.db'
#file = path + '\\fcd\\csv_database_formatted.db' #30 milioni di righe

# If not already present, create the directory to store the downloaded data
results_directory = 'Risultati'
if not os.path.exists(path+'\\'+results_directory):
    os.makedirs(path+'\\'+results_directory)


database = '30000000righe'#'500000righe'
databases_names = glob.glob(path+'databases//'+database+'//*.db')

''' 
	# organizzazione database
	0 - index
	1 - linea
	2 - turno
	3 - date
	4 - mezzo
	5 - lat
	6 - lon
'''

# soglie di visualizzazione velocità
soglia_inf = 0#np.nanmin(array_vel_stats[:,4].astype(np.float64))
soglia_sup = 15#np.nanmax(array_vel_stats[:,4].astype(np.float64))

# per la linea 15 (index 12), linea 38 (index 50), linea 62(index 89) non funziona il grafico
print 'Tot linee:', len(databases_names)

# ciclo sulle linee
for index in xrange(124,125):#(125,126):#len(databases_names)):# index 12 corrisponde a linea 15, problematica
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
        #print  val_mezzo, occ_mezzo
        coord_iso_mezzi = []
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
            #pdb.set_trace()
            iso_mezzo[:,3] = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f') for date in iso_mezzo[:,3]]# [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f').date() for date in iso_mezzo[:,3]]
            #pdb.set_trace()
            date_sorted_iso_mezzo = iso_mezzo[np.argsort(iso_mezzo[:, 3])]
            
            # ciclo per calcolare la velocità (planimetrica)
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
                vel = spost_plan_metres / delta_t.total_seconds() # meters / second
                lon_media = (date_sorted_iso_mezzo[index_time,6] + date_sorted_iso_mezzo[index_time-1,6])/2.0
                lat_media = (date_sorted_iso_mezzo[index_time,5] + date_sorted_iso_mezzo[index_time-1,5])/2.0
                tempo_medio = date_sorted_iso_mezzo[index_time-1,3]+delta_t/2
                #print mezzo, tempo_medio, lat_media, lon_media, vel
                #0: mezzo, 1: tempo_medio, 2:lat_media, 3: lon_media, 4: velocità, 5: delta lat (dy), 6:delta_lon, 7: start lat (start y), 8: start lon (startx), 9: spost_plan-metri, 10:spost_plan_degrees
                vel_stats.append      (np.array([mezzo, tempo_medio, lat_media, lon_media, vel, delta_lat, delta_lon, date_sorted_iso_mezzo[index_time-1,5], date_sorted_iso_mezzo[index_time-1,6], spost_plan_metres, spost_plan_degrees]))
                vel_stats_linea.append(np.array([mezzo, tempo_medio, lat_media, lon_media, vel, delta_lat, delta_lon, date_sorted_iso_mezzo[index_time-1,5], date_sorted_iso_mezzo[index_time-1,6], spost_plan_metres, spost_plan_degrees]))
            #pdb.set_trace()
            coord_iso_mezzi.append(date_sorted_iso_mezzo[:,4:])#mezzo, lat, lon

            '''
            # codice per graficare separatamente la posizione di ogni mezzo di una linea
            x_inf = np.min(date_sorted_iso_mezzo[:,6])-0.01
            x_sup = np.max(date_sorted_iso_mezzo[:,6])+0.01
            y_inf = np.min(date_sorted_iso_mezzo[:,5])-0.01
            y_sup = np.max(date_sorted_iso_mezzo[:,5])+0.01 #45.080

            G = ox.graph_from_bbox(y_sup, y_inf, x_sup, x_inf, network_type='drive', name = 'prova')

            fig, ax = ox.plot_graph(G, close = False, show=False, axis_off=False)

            plt.plot(date_sorted_iso_mezzo[:,6],date_sorted_iso_mezzo[:,5], 'o', color='blue', markeredgecolor='k')
            plt.plot(7.621, 45.074, '*',label = 'capolinea 36', color='yellow', markersize=18, markeredgewidth=2,markeredgecolor='k')
            plt.plot(7.522, 45.070, '*',label = 'capolinea 36', color='red', markersize=18, markeredgewidth=2,markeredgecolor='k')
            plt.xlabel('longitude')
            plt.ylabel('latitude')
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
            fig.savefig(path+'//'+results_directory+'//'+'linea'+linea + '_mezzo'+str(mezzo)+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
            '''
            
            array_vel_stats = np.array(vel_stats)
            if len(array_vel_stats)>0:
                # codice per graficare separatamente lo spostamento di ogni mezzo di una linea
                # allo spostamento assegno la posizione media tra le due posizioni contigue temporalmente
                # il colore è funzione della velocità
                x_inf = np.min(array_vel_stats[:,3])-0.01
                x_sup = np.max(array_vel_stats[:,3])+0.01
                y_inf = np.min(array_vel_stats[:,2])-0.01
                y_sup = np.max(array_vel_stats[:,2])+0.01 #45.080
                # scarico il network stradale della zona da OSM
                G = ox.graph_from_bbox(y_sup, y_inf, x_sup, x_inf, network_type='drive', name = 'prova')

                nz = mcolors.Normalize(vmin = soglia_inf, vmax = soglia_sup)
                fig, ax = ox.plot_graph(G, close = False, show=False, axis_off=False)
                plt.title('Velocities linea '+linea+ '- mezzo '+ str(mezzo))

                #plt.scatter(array_vel_stats[:,3],array_vel_stats[:,2], c =cm.jet(nz(array_vel_stats[:,4].astype(np.float64))), edgecolor='k', s = 35, zorder =5)
                plt.quiver( array_vel_stats[:,8].astype(np.float64), #start x
                    array_vel_stats[:,7].astype(np.float64), #start y
                    array_vel_stats[:,6].astype(np.float64), # /array_vel_stats[:,10].astype(np.float64), # dx #dividendo per il modulo, ottenevo tutte le freccette della stessa lunghezza, ma dovevo scalarle
                    array_vel_stats[:,5].astype(np.float64), # /array_vel_stats[:,10].astype(np.float64), # dy
                    angles='xy', # ‘xy’: arrows point from (x,y) to (x+dx, y+dy). Use this for plotting a gradient field, for example.
                    scale=1, # più è grande, + le frecce sono corte Number of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer. The arrow length unit is given by the scale_units paramete
                    scale_units='xy', # usando le scale units, non è più necessario alterare la scala #to plot vectors in the x-y plane, with u and v having the same units as x and y, use angles='xy', scale_units='xy', scale=1.
                    color=cm.jet(nz(array_vel_stats[:,4].astype(np.float64))),# color = velocities
                    zorder = 5, #più è alto, più il plot è in primo piano
                    edgecolor='k', # colore bordo freccia
                    linewidth=.7,
                    alpha=0.8) # t
                if linea == u'36  ':
                    plt.plot(7.621, 45.074, '*',label = 'capolinea 36', color='yellow', markersize=18, markeredgewidth=2,markeredgecolor='k')
                    plt.plot(7.522, 45.070, '*',label = 'capolinea 36', color='yellow', markersize=18, markeredgewidth=2,markeredgecolor='k')
                plt.xlabel('longitude')
                plt.ylabel('latitude')
                handles, labels = ax.get_legend_handles_labels()
                lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
                cax,_ = mcolorbar.make_axes(plt.gca())
                cb = mcolorbar.ColorbarBase(cax, cmap=cm.jet, norm = nz)
                cb.set_clim(soglia_inf, soglia_sup)
                cb.set_label('velocities [m/s]')
                mng = plt.get_current_fig_manager()
                mng.window.showMaximized()
                fig.savefig(path+'//'+results_directory+'//'+'velocities_linea'+linea + '_mezzo'+str(mezzo)+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
                #plt.show()
                #sys.exit()

        #########################################################################
        # CODICE PER GRAFICARE TUTTI I MEZZI DI UNA LINEA INSIEME
        # primo grafico: grafico della posizione di tutti i mezzi della linea sottoforma di scatterplot il cui colore in funzione dei mezzi
        # secondo grafico: grafico degli spostamenti di tutti i mezzi della linea sottoforma di quiverplot il cui colore è funzione della velocità

        # bounding box che contiene tutte le posizioni dei mezzi che hanno servito quella linea
        x_inf = np.nanmin(iso_linea[:,6])-0.01
        x_sup = np.nanmax(iso_linea[:,6])+0.01
        y_inf = np.nanmin(iso_linea[:,5])-0.01
        y_sup = np.nanmax(iso_linea[:,5])+0.01
        #pdb.set_trace()

        # scarico il network stradale della zona da OSM
        G = ox.graph_from_bbox(y_sup, y_inf, x_sup, x_inf, network_type='drive', name = 'prova')
        #pdb.set_trace()

        # primo grafico: posizione mezzi
        fig, ax = ox.plot_graph(G, close = False, show=False, axis_off=False)
        plt.title('Linea '+linea)
        colors_mezzi = cm.rainbow(np.linspace(0, 1, len(coord_iso_mezzi)))
        for k in xrange(0,len(coord_iso_mezzi)):
            plt.plot(coord_iso_mezzi[k][:,2],coord_iso_mezzi[k][:,1], 'o', color=colors_mezzi[k], label = 'mezzo '+str(coord_iso_mezzi[k][:,0][0]), markeredgecolor='k')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        if linea == u'36  ':
            plt.plot(7.621, 45.074, '*',label = 'capolinea 36', color='yellow', markersize=18, markeredgewidth=2,markeredgecolor='k')
            plt.plot(7.522, 45.070, '*',label = 'capolinea 36', color='yellow', markersize=18, markeredgewidth=2,markeredgecolor='k')
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        fig.savefig(path+'//'+results_directory+'//'+'linea_'+linea +'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')

        # Creo il vettore contenente le informazioni di tutti i mezzi della linea a partire dalla lista globale
        array_vel_stats = np.array(vel_stats_linea)
        
        # secondo grafico: spostamenti e velocità
        fig, ax = ox.plot_graph(G, close = False, show=False, axis_off=False)
        plt.title('Velocities linea '+linea)
        nz = mcolors.Normalize(vmin = soglia_inf, vmax = soglia_sup)
        plt.quiver( array_vel_stats[:,8].astype(np.float64), #start x
                    array_vel_stats[:,7].astype(np.float64), #start y
                    array_vel_stats[:,6].astype(np.float64), # /array_vel_stats[:,10].astype(np.float64), # dx #dividendo per il modulo, ottenevo tutte le freccette della stessa lunghezza, ma dovevo scalarle
                    array_vel_stats[:,5].astype(np.float64), # /array_vel_stats[:,10].astype(np.float64), # dy
                    angles='xy', # ‘xy’: arrows point from (x,y) to (x+dx, y+dy). Use this for plotting a gradient field, for example.
                    scale=1, # più è grande, + le frecce sono corte Number of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer. The arrow length unit is given by the scale_units paramete
                    scale_units='xy', # usando le scale units, non è più necessario alterare la scala #to plot vectors in the x-y plane, with u and v having the same units as x and y, use angles='xy', scale_units='xy', scale=1.
                    color=cm.jet(nz(array_vel_stats[:,4].astype(np.float64))),# color = velocities
                    zorder = 5, #più è alto, più il plot è in primo piano
                    edgecolor='k', # colore bordo freccia
                    linewidth=.7,
                    alpha=0.8) # trasparenza

        if linea == u'36  ':
                plt.plot(7.621, 45.074, '*',label = 'capolinea 36', color='yellow', markersize=18, markeredgewidth=2,markeredgecolor='k')
                plt.plot(7.522, 45.070, '*',label = 'capolinea 36', color='yellow', markersize=18, markeredgewidth=2,markeredgecolor='k')
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
        cax,_ = mcolorbar.make_axes(plt.gca())
        cb = mcolorbar.ColorbarBase(cax, cmap=cm.jet, norm = nz)
        cb.set_clim(soglia_inf, soglia_sup)
        cb.set_label('velocities [m/s]')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        fig.savefig(path+'//'+results_directory+'//'+'velocities_linea_'+linea +'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
        iso_linea = None


print '   vel min', np.min(array_vel_stats[:,4].astype(np.float64)), 'm/s , ', np.min(array_vel_stats[:,4].astype(np.float64)) * 3600 / 1000.0, 'km/h'
print '   vel max', np.max(array_vel_stats[:,4].astype(np.float64)), 'm/s , ', np.max(array_vel_stats[:,4].astype(np.float64)) * 3600 / 1000.0, 'km/h'
print ' vel media', np.nanmean(array_vel_stats[:,4].astype(np.float64)), 'm/s , ', np.mean(array_vel_stats[:,4].astype(np.float64)) * 3600 / 1000.0, 'km/h'
print 'vel median', np.median(array_vel_stats[:,4].astype(np.float64)), 'm/s , ', np.median(array_vel_stats[:,4].astype(np.float64)) * 3600 / 1000.0, 'km/h'

print 'Database read in', (dt.datetime.now() - start).seconds, 'seconds'
plt.show()
