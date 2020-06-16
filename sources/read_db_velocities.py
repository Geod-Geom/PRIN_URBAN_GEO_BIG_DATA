#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
    Script name: read_db_velocities.py
    Description: the script analyzes and plot the data contained in the 
                 previously computed databases of the velocities
    Authors: Roberta Ravanelli   <roberta.ravanelli@uniroma1.it>,
             Geodesy and Geomatics Division (DICEA) @ University of Rome "La Sapienza"
    Python Version: 2.7
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

plt.rcParams['text.latex.preamble'] = [r'\boldmath']# tutte le scitte in grassetto
plt.rcParams.update({'font.size': 25})
plt.rc('text', usetex=False)


plt.switch_backend('Qt5Agg')
print '#1 Backend:',plt.get_backend()

#np.set_printoptions(threshold=np.set_printoptions(threshold=np.nan))# evito che quando printo a schermmo, mi metta i ...https://stackoverflow.com/questions/1987694/print-the-full-numpy-array


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


def boh(drive_network_graph, start_lon, start_lat, delta_lon, delta_lat, velocities, soglia_inf = 0, soglia_sup= 80):
    fig_from_function, ax_from_function = ox.plot_graph(drive_network_graph, close = False, show=False, axis_off=False)
    nz = mcolors.Normalize(vmin = soglia_inf, vmax = soglia_sup)
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.quiver( start_lon, # start x
                start_lat, # start y
                delta_lon, # delta x 
                delta_lat, # delta y
                angles='xy', # ‘xy’: arrows point from (x,y) to (x+dx, y+dy). Use this for plotting a gradient field, for example.
                scale=1, # più è grande, + le frecce sono corte Number of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer. The arrow length unit is given by the scale_units paramete
                scale_units='xy', # usando le scale units, non è più necessario alterare la scala #to plot vectors in the x-y plane, with u and v having the same units as x and y, use angles='xy', scale_units='xy', scale=1.
                color=cm.jet(nz(velocities)),# color = velocities
                zorder = 5, #più è alto, più il plot è in primo piano
                edgecolor='k', # colore bordo freccia
                linewidth=.7,
                alpha=0.8) # trasparenza
    if linea == u'36  ':
                plt.plot(7.621, 45.074, '*',label = 'capolinea 36', color='yellow', markersize=18, markeredgewidth=2,markeredgecolor='k')
                plt.plot(7.522, 45.070, '*',label = 'capolinea 36', color='yellow', markersize=18, markeredgewidth=2,markeredgecolor='k')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    handles, labels = ax_from_function.get_legend_handles_labels()
    lgd = ax_from_function.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
    cax,_ = mcolorbar.make_axes(plt.gca())
    cb = mcolorbar.ColorbarBase(cax, cmap=cm.jet, norm = nz)
    cb.set_clim(soglia_inf, soglia_sup)
    cb.set_label('velocities [km/h]')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    #fig_from_function.savefig(path+'//'+results_directory+'//'+'velocities_linea_'+linea +'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()
    return fig_from_function, ax_from_function

start = dt.datetime.now()

path = os.path.dirname(os.path.abspath(__file__))+'//..//' #esco dalla sub directory codice

# If not already present, create the directory to store the downloaded data
results_directory = 'RisultatiVelocities_prova'
if not os.path.exists(path+'\\'+results_directory):
    os.makedirs(path+'\\'+results_directory)

fasce_orarie = np.array(([0,5],[5,7],[7,9],[9,11],[11,13],[13,15],[15,17],[17,19],[19,21],[21,24]))

database = 'velocities_nuove'#'500000righe'
databases_names = glob.glob(path+'databases//'+database+'//*.db')
#sys.exit()
'''
# organizzazione database
0 mezzo BIGINT
1 tempo_medio DATETIME
2 lat_media FLOAT
3 lon_media FLOAT
4 vel_m_s FLOAT
5 delta_lat FLOAT
6 delta_lon FLOAT
7 start_lat FLOAT
8 start lon FLOAT
9 spost_plan_metri FLOAT
10 spost_plan_degrees FLOAT
11 delta_t_seconds FLOAT
12 start_time DATETIME
13 weekday BIGINT
'''

# per la linea 15 (index 12), linea 38 (index 50), linea 62(index 89) non funziona il grafico
print 'Tot linee:', len(databases_names)

#linea 11, index 5
#linea 12, index 7
#linea 13, index 8
#linea 39, index 51
# ciclo sulle linee
for index in xrange(0, 1):#len(databases_names)):# index 12 corrisponde a linea 15, problematica
    linea = databases_names[index][len(path+'databases//'+database+'//csv_database_velocities'):-3]
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
        # i dati di ogni singola linea sono già ordinati per mezzo
        iso_linea[:,4] = iso_linea[:,4].astype(np.float64) * 3600 / 1000.0 # velocità in km/s
        #iso_linea[:,1] = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f') for date in iso_linea[:,1].astype(str)]
        #iso_linea[:,1] = iso_linea[:,1].astype('datetime64[ms]')
        # rimuovo le righe vuote (None o nan)
        
        #mask_outlier = np.logical_or (iso_linea[:,4] == np.inf , iso_linea[:,1] == np.array(None), iso_linea[:,4].astype(np.float64)>soglia_outlier))
        
        # rimuovo le velocità infinite
        '''indici_outlier = np.where(iso_linea[:,4].astype(np.float64)== np.inf)[0]
        if len (indici_outlier)>0:
            print len(indici_outlier), len(iso_linea), len(iso_linea) - len(indici_outlier)
            iso_linea = np.delete(iso_linea, indici_outlier, axis=0)'''

        # STATISTICHE E GRAFICO SUI TUTTI I DATI (QUINDI CONTENGONO ANCHE GLI OUTLIER)

        # bounding box che contiene tutte le posizioni dei mezzi che hanno servito quella linea
        x_inf_tot = np.nanmin(iso_linea[:,3].astype(np.float64))-0.01
        x_sup_tot = np.nanmax(iso_linea[:,3].astype(np.float64))+0.01
        y_inf_tot = np.nanmin(iso_linea[:,2].astype(np.float64))-0.01
        y_sup_tot = np.nanmax(iso_linea[:,2].astype(np.float64))+0.01

        # scarico il network stradale della zona da OSM
        G_tot = ox.graph_from_bbox(y_sup_tot, y_inf_tot, x_sup_tot, x_inf_tot, network_type='drive', name = 'prova')

        #boh(drive_network_graph, start_lon, start_lat, delta_lon, delta_lat, velocities):
        fig_tot, ax_tot = boh(G_tot, iso_linea[:,8].astype(np.float64), iso_linea[:,7].astype(np.float64), iso_linea[:,6].astype(np.float64), iso_linea[:,5].astype(np.float64), iso_linea[:,4].astype(np.float64))
        ax_tot.set_title('Velocities linea '+linea+' all data')
        fig_tot.canvas.set_window_title('Velocities linea '+linea+' all data')
        fig_tot.savefig(path+'//'+results_directory+'//'+'velocities_linea_'+linea +'all_data.png', bbox_inches='tight')#bbox_extra_artists=(lgd,)

        print '**** STATISTICHE GLOBALI (outlier compresi) ****'
        print '   vel min', format(np.min(np.ma.masked_invalid(iso_linea[:,4].astype(np.float64))),'.2f'), 'km/h'
        print '   vel max', format(np.max(np.ma.masked_invalid(iso_linea[:,4].astype(np.float64))),'.2f'), 'km/h'
        print ' vel media', format(np.mean(np.ma.masked_invalid(iso_linea[:,4].astype(np.float64))),'.2f'), 'km/h'
        print 'vel median', format(np.nanmedian(np.ma.masked_invalid(iso_linea[:,4].astype(np.float64))),'.2f'), 'km/h'
        print '   vel std', format(np.std(np.ma.masked_invalid(iso_linea[:,4].astype(np.float64))), '.2f'),    'km/h'

        print '   min dt',  format(np.nanmin(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'
        print '   max dt',  format(np.nanmax(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'
        print '  media dt',  format(np.nanmean(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'
        print ' median dt',  format(np.nanmedian(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'
        print '    std dt', format(np.nanstd(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'

        # as type float and as type int trasforma i none in -2147483648
        dt_val_e_occ = find_histogram_class_from_int(iso_linea[:,11].astype(np.float64).astype(np.int))

        print 'histo dt linea', linea
        print dt_val_e_occ[0]

        plt.figure('histo dt linea '+ str(linea))
        plt.bar(dt_val_e_occ[0][:,0], dt_val_e_occ[0][:,1], align='center', color='blue')
        plt.suptitle('histo dt linea '+ str(linea))
        plt.xlabel('dt values [s]')
        plt.ylabel('occurrencies [-]')
        plt.xlim(0, 5000)#np.max(dt_val_e_occ[0][:,0]))##len(dt_val_e_occ[0][:,0])) #grafico solo i dt fino a 300 secondi (5 minuti)
        plt.ylim(0, np.max(dt_val_e_occ[0][:,1]))
        plt.grid()
        plt.savefig(path+results_directory+'//'+'histo_dt_linea_'+str(linea)+'.png')


        '''
        mask_tempi_con_solo_una_occorenza = dt_val_e_occ[0][:,1]>1
        val_dt_occ_maggiori_1 = dt_val_e_occ[0][mask_tempi_con_solo_una_occorenza][:,0]# valori di dt con numero di occorrenze > 1
        val_dt_occ_maggiori_1 = val_dt_occ_maggiori_1[val_dt_occ_maggiori_1>2]# elimino i dt negativi (as type int trasforma i none in -2147483648), compresi i valori con dt<2

        ## questo ciclo fa una sorta di np.where quando l'array viene comparato a un altrp array (e non a uno scalare)
        maschera = np.zeros(len(iso_linea))*False # 0 corrisponde a false
        for indice in xrange (len(val_dt_occ_maggiori_1)):
            maschera_val = iso_linea[:,11] == val_dt_occ_maggiori_1[indice]
            maschera = np.logical_or(maschera,maschera_val)
        print maschera
        # con questa maschera conservo esclusivamente quei valori di iso_linea
        # i cui dt occorrono più di una volta
        iso_linea =iso_linea[maschera]'''

        dt_cut_off_sup = np.percentile(iso_linea[:,11], 99.5)
        dt_cut_off_inf = np.percentile(iso_linea[:,11], 0.5)
        vel_cut_off_sup = np.percentile(iso_linea[:,4], 99.5)
        vel_cut_off_inf = np.percentile(iso_linea[:,4], 0.5)
        print 'dt cut off inf', dt_cut_off_inf
        print 'dt cut off sup', dt_cut_off_sup
        print 'vel cut off inf', vel_cut_off_inf
        print 'vel cut off sup', vel_cut_off_sup
        
        iso_linea = iso_linea[iso_linea[:,11] < dt_cut_off_sup]
        iso_linea = iso_linea[iso_linea[:,11] > dt_cut_off_inf]

        soglia_outlier = 5 * np.mean(np.ma.masked_invalid(iso_linea[:,4].astype(np.float64)))
        print 'vel soglia outlier', soglia_outlier

        #https://stackoverflow.com/questions/20528328/numpy-logical-or-for-more-than-two-arguments
        # trovo gli indici che hanno valori problematici (None o inf) o maggiori della soglia outlier o velocità nulle
        #mask_outlier2 = np.logical_or.reduce( (iso_linea[:,4] == np.inf , iso_linea[:,1] == np.array(None), iso_linea[:,4] < 0.1, iso_linea[:,4].astype(np.float64)>soglia_outlier))
        mask_outlier2 = np.logical_or.reduce( ( iso_linea[:,4] < 0.1, iso_linea[:,4].astype(np.float64)>soglia_outlier))
        
        #mask_outlier2 = mask_outlier2 = np.logical_or.reduce( (iso_linea[:,4] == np.inf , iso_linea[:,1] == np.array(None), iso_linea[:,4] < 0.1))
        '''
        mask_outlier = (iso_linea[:,1] == np.array(None)) | (iso_linea[:,4].astype(np.float64)> soglia_outlier) | (iso_linea[:,4] == np.inf)|  (iso_linea[:,4] < 0.1)

        # posso usare indifferentemente il bitwise or o il logical or (ho controllato e i risultati sono uguali)
        print mask_outlier == mask_outlier2
        print np.max(mask_outlier == mask_outlier2), np.min(mask_outlier == mask_outlier2)
        print 'elementi true',np.count_nonzero(mask_outlier2), np.count_nonzero(mask_outlier2)# conto i true
        '''
        # rimuovo i record classificati come outlier (in base alle velocità)
        indici_outlier = np.where(mask_outlier2)[0]
        if len (indici_outlier)>0:
            print len(indici_outlier), len(iso_linea), len(iso_linea) - len(indici_outlier)
            iso_linea = np.delete(iso_linea, indici_outlier, axis=0)
        print len(indici_outlier), len(iso_linea)

        
        dt_val_e_occ_cleaned = find_histogram_class_from_int(iso_linea[:,11].astype(np.float64).astype(np.int))
        print dt_val_e_occ_cleaned

        plt.figure('histo dt cleaned linea'+ str(linea))
        plt.bar(dt_val_e_occ_cleaned[0][:,0], dt_val_e_occ_cleaned[0][:,1], align='center', color='blue')
        plt.suptitle('histo dt linea '+ str(linea))
        plt.xlabel('dt values [s]')
        plt.ylabel('occurrencies [-]')
        plt.grid()
        plt.xlim(0, np.max(dt_val_e_occ_cleaned[0][:,0]))#300)#len(dt_val_e_occ_cleaned[0][:,0])) #grafico solo i dt fino a 300 secondi (5 minuti)
        plt.ylim(0, np.max(dt_val_e_occ_cleaned[0][:,1]))
        plt.savefig(path+results_directory+'//'+'histo_dt_cleaned_linea_'+str(linea)+'.png')


        print '**** STATISTICHE SENZA OUTLIER ****'
        print '   vel min', format(np.min(iso_linea[:,4].astype(np.float64)),'.2f'), 'km/h'
        print '   vel max', format(np.max(iso_linea[:,4].astype(np.float64)),'.2f'), 'km/h'
        print ' vel media', format(np.mean(iso_linea[:,4].astype(np.float64)),'.2f'), 'km/h'
        print 'vel median', format(np.median(iso_linea[:,4].astype(np.float64)),'.2f'), 'km/h'
        print '   vel std', format(np.std(iso_linea[:,4].astype(np.float64)), '.2f'),    'km/h'

        print '   min dt',  format(np.nanmin(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'
        print '   max dt',  format(np.nanmax(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'
        print '  media dt',  format(np.nanmean(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'
        print ' median dt',  format(np.nanmedian(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'
        print '    std dt', format(np.nanstd(iso_linea[:,11].astype(np.float64)), '.2f'), 'seconds'

        soglia_inf = np.int(np.floor(np.min(iso_linea[:,4].astype(np.float64))))#km/h
        soglia_sup = np.int(np.ceil(np.max(iso_linea[:,4].astype(np.float64)))) #km/h#15 m/s

        fig_tot_no_outlier, ax_tot_no_outlier = boh(G_tot, iso_linea[:,8].astype(np.float64), iso_linea[:,7].astype(np.float64), iso_linea[:,6].astype(np.float64), iso_linea[:,5].astype(np.float64), iso_linea[:,4].astype(np.float64))
        ax_tot_no_outlier.set_title('Velocities linea '+linea+' no outlier')
        fig_tot_no_outlier.canvas.set_window_title('Velocities linea '+linea+' no outlier')
        fig_tot_no_outlier.savefig(path+'//'+results_directory+'//'+'velocities_linea_'+linea +'no_outlier.png', bbox_inches='tight')
        #plt.show()
        #sys.exit()


        #Inizio analisi giorni festivi-feriali
        mask_lavorativi = iso_linea[:,-1]<5
        mask_weekend =  iso_linea[:,-1]>4

        masks_feriali_festivi = np.array([mask_lavorativi, mask_weekend])
        labels_work_weekend= np.array(['lavorativi', 'festivi'])
        
        
        
        
        
        for indice_feriali_festivi in xrange (len(masks_feriali_festivi)):
            iso_linea_appoggio = iso_linea.copy()

            print "******** GIORNI", labels_work_weekend[indice_feriali_festivi], "********"
            iso_linea_work_weekend = iso_linea_appoggio[masks_feriali_festivi[indice_feriali_festivi]]
            print len(iso_linea_work_weekend)

            # controllo singoli giorni (festivi e lavorativi)
            df_date = pd.DataFrame(iso_linea_work_weekend)
            giorni = np.array(pd.to_datetime(df_date[12]).dt.day)
            giorni_val_e_occ = find_histogram_class_from_int(giorni)
            print giorni_val_e_occ
            '''
            for indice_giorno in range(len(giorni_val_e_occ[0][:,0])):
                iso_linea_appoggio_sabati_domeniche = iso_linea_work_weekend.copy()
                print giorni_val_e_occ[0][indice_giorno,0]
                
                print 'vel giorno ', giorni_val_e_occ[0][indice_giorno,0], 
                mask_leave_one_out = giorni == giorni_val_e_occ[0][indice_giorno,0]
                iso_linea_appoggio_sabati_domeniche = iso_linea_appoggio_sabati_domeniche[mask_leave_one_out]
                dfg = pd.DataFrame(iso_linea_appoggio_sabati_domeniche)
                dfg[12] = pd.to_datetime(dfg[12])
                giorni_bis = np.array(dfg[12].dt.day)
                print find_histogram_class_from_int(giorni_bis)
                print np.mean(iso_linea_appoggio_sabati_domeniche[:,4]), np.median(iso_linea_appoggio_sabati_domeniche [:,4]), np.std(iso_linea_appoggio_sabati_domeniche [:,4])

            
                #iso_linea_work_weekend = iso_linea_appoggio_sabati_domeniche
            
                # INIZIO ANALISI TEMPORALE: SUDDIVISIONE IN FASCE ORARIE
                #converto le stringhe in date time objects
                iso_linea_appoggio_sabati_domeniche[:,1] = iso_linea_appoggio_sabati_domeniche[:,1].astype('datetime64[ms]')
                iso_linea_appoggio_sabati_domeniche[:,12] = iso_linea_appoggio_sabati_domeniche[:,12].astype('datetime64[ms]')
                #dates = pd.DatetimeIndex(iso_linea_work_weekend[:,1])# old: tempo medio
                dates = pd.DatetimeIndex(iso_linea_appoggio_sabati_domeniche[:,12])# new: start time

                stats_vel_fasce = []
                for t  in xrange(0,len(fasce_orarie)):
                    print '\nFascia oraria', fasce_orarie[t,0], ' - ', fasce_orarie[t,1],':'

                    # individuo i record acquisiti nella specifica fascia oraria
                    indici_fascia = np.where(np.logical_and( dates.hour >= fasce_orarie[t,0], dates.hour < fasce_orarie[t,1]))[0]
                    dati_fascia = iso_linea_appoggio_sabati_domeniche[indici_fascia]

                    print len(dati_fascia), 'record\n'

                    if len(dati_fascia) > 0:
                                #print indici_fascia

                                min_vel    = np.min    (dati_fascia[:,4])
                                max_vel    = np.max    (dati_fascia[:,4])
                                mean_vel   = np.mean   (dati_fascia[:,4])
                                median_vel = np.median (dati_fascia[:,4])
                                std_vel    = np.std    (dati_fascia[:,4])

                                stats_vel_fasce.append(np.array([fasce_orarie[t,0], fasce_orarie[t,1], str(fasce_orarie[t,0]) +' - '+ str(fasce_orarie[t,1]), min_vel, max_vel, mean_vel, median_vel, std_vel, len(dati_fascia)]))

                                print '   vel min', format( min_vel,    '.2f'),    'km/h'
                                print '   vel max', format( max_vel,    '.2f'),    'km/h'
                                print ' vel media', format( mean_vel,   '.2f'),   'km/h'
                                print 'vel median', format( median_vel, '.2f'), 'km/h'
                                print '   vel std', format( std_vel,    '.2f'),    'km/h'
                                print
                                print '   min dt',  format(np.nanmin(dati_fascia[:,11].astype(np.float64)),     '.2f'), 'seconds'
                                print '   max dt',  format(np.nanmax(dati_fascia[:,11].astype(np.float64)),     '.2f'), 'seconds'
                                print '  media dt', format(np.nanmean(dati_fascia[:,11].astype(np.float64)),    '.2f'), 'seconds'
                                print ' median dt', format(np.nanmedian(dati_fascia[:,11].astype(np.float64)),  '.2f'), 'seconds'
                                print '    std dt', format(np.nanstd(dati_fascia[:,11].astype(np.float64)),     '.2f'), 'seconds'
                                print

                                x_inf_fascia = np.nanmin(dati_fascia[:,3].astype(np.float64))-0.01
                                x_sup_fascia = np.nanmax(dati_fascia[:,3].astype(np.float64))+0.01
                                y_inf_fascia = np.nanmin(dati_fascia[:,2].astype(np.float64))-0.01
                                y_sup_fascia = np.nanmax(dati_fascia[:,2].astype(np.float64))+0.01
                                G_fascia = ox.graph_from_bbox(y_sup_fascia, y_inf_fascia, x_sup_fascia, x_inf_fascia, network_type='drive', name = 'prova')

                                fig_fascia,ax_fascia = boh(G_fascia, dati_fascia[:,8].astype(np.float64), dati_fascia[:,7].astype(np.float64), dati_fascia[:,6].astype(np.float64), dati_fascia[:,5].astype(np.float64), dati_fascia[:,4].astype(np.float64), soglia_inf, soglia_sup)
                                fig_fascia.canvas.set_window_title('Velocities linea '+linea+' fascia oraria '+ str(fasce_orarie[t,0])+ ' - '+ str(fasce_orarie[t,1]) +' giorni '+ labels_work_weekend[indice_feriali_festivi])
                                ax_fascia.set_title('Velocities linea '+linea+' fascia oraria '+ str(fasce_orarie[t,0])+ ' - '+ str(fasce_orarie[t,1])+' giorni '+ labels_work_weekend[indice_feriali_festivi])
                                fig_fascia.savefig(path+'//'+results_directory+'//giorni_'+labels_work_weekend[indice_feriali_festivi]+'_velocities_linea_'+linea +'_fascia_oraria_'+ str(fasce_orarie[t,0])+ '_'+ str(fasce_orarie[t,1])+'giorno_'+str(giorni_val_e_occ[0][indice_giorno,0])+'.png', bbox_inches='tight')

                array_stats_vel_fasce = np.array(stats_vel_fasce)
                stats_vel_fasce = None

                fig =plt.figure("Vel stats linea"+linea+' giorni '+ labels_work_weekend[indice_feriali_festivi])
                ax = fig.add_subplot(111)
                ax.set_title('Velocities linea '+linea+' giorni '+ labels_work_weekend[indice_feriali_festivi])
                x_ticks_labels = array_stats_vel_fasce[:,2]
                plt.plot(array_stats_vel_fasce[:,0],array_stats_vel_fasce[:,5],'bo--',label='vel mean',  markersize=10, markeredgewidth=2,markeredgecolor='k')
                plt.plot(array_stats_vel_fasce[:,0],array_stats_vel_fasce[:,6],'ro--',label='vel median',  markersize=10, markeredgewidth=2,markeredgecolor='k')
                # Set ticks labels for x-axis
                plt.xticks(array_stats_vel_fasce[:,0].astype(float), x_ticks_labels, rotation='vertical')
                plt.legend()#loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, fancybox=False, shadow=False, frameon=False)
                plt.ylabel('Velocity [km/h]')
                plt.xlabel('Fasce orarie [h]')
                plt.grid()
                mng = plt.get_current_fig_manager()
                mng.window.showMaximized()
                #fig.savefig(path+'//'+results_directory+'//giorni_'+labels_work_weekend[indice_feriali_festivi]+'_velocities_linea_'+linea +'_fasce_orarie.png')
                #iso_linea = None
                #sys.exit()
                with open(path+'//'+results_directory+'//giorni_'+labels_work_weekend[indice_feriali_festivi]+'_velocities_stats_linea_'+linea +'_fasce_orarie_giorno'+str(giorni_val_e_occ[0][indice_giorno,0])+'.txt',"w") as file_stats:
                    file_stats.write('start hour[h]\tend hour[h]\tnumerosità [-]\tmin vel[km/h]\tmax vel [km/h]\tmean vel [km/h]\tmedian vel [km/h]\t std vel [km/h]\n')
                    for ind in xrange(len(array_stats_vel_fasce)):
                        file_stats.write(str(array_stats_vel_fasce[ind,0])+'\t'+str(array_stats_vel_fasce[ind,1])+'\t'+str(array_stats_vel_fasce[ind,8])+'\t'+str(array_stats_vel_fasce[ind,3])+'\t'+str(array_stats_vel_fasce[ind,4])+'\t'+str(array_stats_vel_fasce[ind,5])+'\t'+str(array_stats_vel_fasce[ind,6])+'\t'+str(array_stats_vel_fasce[ind,7])+'\n')
                #sys_exit()
                '''
plt.show()


print 'Database read in', (dt.datetime.now() - start).seconds, 'seconds'