# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/41176167/animate-matplotlib-plot-just-once-and-choose-different-range-for-frames
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math
import os
from pyproj import Proj, transform
from matplotlib import rc,rcParams
from matplotlib.widgets import Button
import pdb
from datetime import datetime
from datetime import timedelta
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
plt.close("all")
plt.switch_backend("Qt5Agg")

def read_db (filename, tablename='table'):
    import sqlite3 as lite
    con = lite.connect(filename)
    with con:
        cur = con.cursor()
        cur.execute('PRAGMA table_info('+tablename+')')
        metadata = cur.fetchall()

        for d in metadata:
            print d[0], d[1], d[2]
        # Dictionary cursor: we can refer to the data by their column names
        con.row_factory = lite.Row

        cur = con.cursor() 
        cur.execute("SELECT * FROM "+tablename)

        rows = cur.fetchall()
        # il problema era che i dati dentro iso_linea non sono più numeri, ma stringhe, 
        # per cui la maschera con un numero non funziona più
        # il dtype è fondamentale per evitare questo problema
        return np.asarray(rows, dtype = object)

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

def ds(E_avanti, N_avanti, E_indietro, N_indietro ):
        ds = np.sqrt((E_avanti-E_indietro)**2+ (N_avanti-N_indietro)**2)
        return ds

path = os.path.dirname(os.path.abspath(__file__))+'//..//'
line = 39
mezzo = 2581#309,363,2581,2584, 8265
data_stops = read_db(path+'databases//routes_databases//routes_database_'+str(line)+'.db','route_table')
# conversion to web mercator coordinates (E,N from phi and lambda)
# https://gis.stackexchange.com/questions/247871/convert-gps-coordinates-to-web-mercator-epsg3857-using-python-pyproj
for index in range (len(data_stops)):
    data_stops[index, 0], data_stops[index, 1] = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), data_stops[index, 0], data_stops[index, 1])

results = np.genfromtxt (path+'//Risultati_proiezione//linea_'+str(line)+'_mezzo_'+str(mezzo)+'.txt', skip_header =1, dtype = object)
#results = results[40:, :]
results[:,0:3] = results[:,0:3].astype(np.float)
results[:, -4:] = results[:, -4:].astype(np.float)
#rimuovo gli outlier
results = results[~np.isnan(results[:,1].astype(float))]
#results = results[:4,:]
print results
#https://stackoverflow.com/questions/9958506/element-wise-string-concatenation-in-numpy/13906671
time = results[:,3]+ results[:,4]
E_proj = results[:,1]
N_proj = results[:,2]
time = [datetime.strptime(date, '%Y/%m/%d%H:%M:%S') for date in time]

data_for_quiver_plot = np.zeros((len(results)-1,5))
list_for_quiver_plot = []
for i in range (1, len(results)):
    # i punto avanti (FCD riproiettato)
    # i-1 punto indietro
    d_s = np.sqrt((E_proj[i] - E_proj[i-1])**2 + (N_proj[i] - N_proj[i-1])**2)
    dt = time [i] - time [i-1]
    v_old = d_s / dt.total_seconds()# m/s 
    print '\nPunti ', int(results[i-1,0]), int(results[i,0]) #i-1, i# dopo aver rimosso gli outlier, l'ID del punto non corrsiponde più 1 a 1 con l'indice del ciclo
    print "Nodi ", int(results[i-1,5]), int(results[i,5])
    if int(results[i,5])!= int(results[i-1,5]):
        # accrocco orribile per immagazzinare la v_new, che scopro solo alla fine
        # cioè quando ho calcolato tutti i ds
        temporary_list = []
        delta_s = 0

        
        
        print  'P'+str(int(results[i-1,0])), '(N'+str(int(results[i-1,5]))+')','-> N'+str(int(results[i-1,5])+1)
        delta_s = delta_s + ds(data_stops[int(results[i-1,5])+1,0], data_stops[int(results[i-1,5])+1,1], E_proj[i-1], N_proj[i-1] )
        temporary_list.append( [results[i-1,0], results[i,0], results[i-1,5], results[i-1,5]+1, E_proj[i-1], N_proj[i-1], data_stops[int(results[i-1,5])+1,0], data_stops[int(results[i-1,5])+1,1], v_old, ds(data_stops[int(results[i-1,5])+1,0], data_stops[int(results[i-1,5])+1,1], E_proj[i-1], N_proj[i-1] )])
        #print  range(int(results[i-1,5]) + 2, int(results[i,5])+1) 
        #print  range(int(results[i-1,5]) + 1, int(results[i,5]))
        #number_of_internal_nodes = 
        # ciclo sui nodi
        for index in range(int(results[i-1,5]) + 1, int(results[i,5])):
            print 'N'+str(index), '->', 'N'+str(index +1)
            
            #list_for_quiver_plot.append( [data_stops[index, 0], data_stops[index, 1], data_stops[index+1, 0], data_stops[index+1, 1], v_old]  )
            delta_s = delta_s + ds( data_stops[index+1, 0], data_stops[index+1, 1], data_stops[index, 0], data_stops[index, 1])
            temporary_list.append( [ results[i-1,0], results[i,0], index, index+1, data_stops[index, 0], data_stops[index, 1], data_stops[index+1, 0], data_stops[index+1, 1], v_old, ds( data_stops[index+1, 0], data_stops[index+1, 1], data_stops[index, 0], data_stops[index, 1])]  )

        print  'N'+str(int(results[i,5])), '->' , 'P'+str(int(results[i,0])), '(N'+str(int(results[i,5]))+')'
        
        
        # io ho un dt spalmato su tanti ds
        # assumo che in questi tratti il bus si sia spostato con velocità costante
        delta_s = delta_s + ds(E_proj[i], N_proj[i], data_stops[int(results[i,5]),0], data_stops[int(results[i,5]),1])
        temporary_list.append( [results[i -1,0], results[i,0], results[i,5], results[i,5], data_stops[int(results[i,5]),0], data_stops[int(results[i,5]),1],     E_proj[i], N_proj[i], v_old, ds(E_proj[i], N_proj[i], data_stops[int(results[i,5]),0], data_stops[int(results[i,5]),1])] )
        v_new = (delta_s) /(dt.total_seconds())# m/s
        temporary_list = np.array(temporary_list)
        temporary_list[:,-2] = v_new# sostituisco v_old con v_new
        delta_time = timedelta(seconds=0)
        for kkkk in range(len(temporary_list)):
                # devo appendere riga per riga perchè tutto insieme non è fattibile (Python si arrabbia)
                #print temporary_list[kkkk, 0], temporary_list[kkkk, 1],temporary_list[kkkk, 2], temporary_list[kkkk, 3], temporary_list[kkkk, 4], temporary_list[kkkk, 5], temporary_list[kkkk, 6]
                dt_punto_indietro = delta_time # delta time of punto indietro
                delta_time = delta_time + timedelta(seconds=temporary_list[kkkk,-1]/ v_new) # delta time of punto avanti
                list_for_quiver_plot.append([temporary_list[kkkk, 0], temporary_list[kkkk, 1],temporary_list[kkkk, 2], temporary_list[kkkk, 3], time [i-1]+dt_punto_indietro, temporary_list[kkkk, 4], temporary_list[kkkk, 5], time [i-1]+delta_time, temporary_list[kkkk, 6], temporary_list[kkkk, 7], temporary_list[kkkk, 8]])
                #print  time [i-1]+delta_time,
                print time [i-1]+dt_punto_indietro, time [i-1]+delta_time#timedelta(seconds=temporary_list[kkkk,-1]/v_new).seconds
                #pdb.set_trace()
        print time [i-1], time [i]#
        #print time [i]#time [i-1]+v_new * temporary_list[:,-1] # dt in seconds per ogni 
        
        #pdb.set_trace()    #print 
        print v_new
    else:
        #list_for_quiver_plot.append( [E_proj[i-1], N_proj[i-1], E_proj[i] - E_proj[i-1], N_proj[i] - N_proj[i-1], v_old])
        list_for_quiver_plot.append( [results[i-1, 0], results[i,0], results[i-1,5], results[i,5], time[i-1], E_proj[i-1], N_proj[i-1], time[i], E_proj[i], N_proj[i], v_old])
        print  'P'+str(int(results[i-1,0])), '(N'+str(int(results[i-1,5]))+')' '->' , 'P'+str(int(results[i,0])), '(N'+str(int(results[i,5]))+')'
        print int(results[i-1,5]), int(results[i,5])
        print time[i-1], time [i]
        print v_old
        
    '''data_for_quiver_plot[i-1, 0] = E_proj[i-1]# start x
    data_for_quiver_plot[i-1, 1] = N_proj[i-1]# start y
    data_for_quiver_plot[i-1, 2] = E_proj[i] - E_proj[i-1]# delta x
    data_for_quiver_plot[i-1, 3] = N_proj[i] - N_proj[i-1]# delta y
    data_for_quiver_plot[i-1, 4] = v_old # velocities'''

data_for_quiver_plot = np.array(list_for_quiver_plot)


# m/s -> km/h
data_for_quiver_plot[:,-1] = data_for_quiver_plot[:,-1]*3.6

soglia_inf = 0
soglia_sup = 80# km/h
nz = mcolors.Normalize(vmin = soglia_inf, vmax = soglia_sup)


fig = plt.figure('Line ' +str(line)+' bus '+str(mezzo))
plt.title('Line ' +str(line)+' bus n. '+str(mezzo), fontweight='bold')
plt.grid()
plt.plot(data_stops[:,0],data_stops[:,1], 'ko--')
#plt.plot(E_proj,N_proj, 'ro' )#'rx', markersize= 12)
ax = fig.add_subplot(111)

plt.quiver(     data_for_quiver_plot[:,5].astype(float), # start x
                data_for_quiver_plot[:,6].astype(float), # start y
                (data_for_quiver_plot[:,8]-data_for_quiver_plot[:,5]).astype(float), # delta x 
                (data_for_quiver_plot[:,9]-data_for_quiver_plot[:,6]).astype(float), # delta y
                angles='xy', # ‘xy’: arrows point from (x,y) to (x+dx, y+dy). Use this for plotting a gradient field, for example.
                scale=1, # più è grande, + le frecce sono corte Number of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer. The arrow length unit is given by the scale_units paramete
                scale_units='xy', # usando le scale units, non è più necessario alterare la scala #to plot vectors in the x-y plane, with u and v having the same units as x and y, use angles='xy', scale_units='xy', scale=1.
                color=plt.cm.jet(nz(data_for_quiver_plot[:, -1].astype(float))),# color = velocities
                zorder = 5, #più è alto, più il plot è in primo piano
                edgecolor='k', # colore bordo freccia
                linewidth=.7,
                alpha=0.8) # trasparenza

plt.xlabel('E [m]', fontweight='bold')
plt.ylabel('N [m]', fontweight='bold')
for zz in range(len(data_stops)):
        plt.text(data_stops[zz,0]-0.00001, data_stops[zz,1]-0.00001, str(zz) , color='k',fontsize=12 )
plt.axis('equal')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.1))
cax,_ = mcolorbar.make_axes(plt.gca())
cb = mcolorbar.ColorbarBase(cax, cmap=cm.jet, norm = nz)
cb.set_clim(soglia_inf, soglia_sup)
cb.set_label('velocities [km/h]', fontweight='bold')#cb.set_label('velocities [m/s]')
#plt.ticklabel_format(useOffset=False, style='plain')
#plt.tight_layout()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.savefig(path+'Risultati_proiezione//'+'Line_' +str(line)+'_bus_'+str(mezzo)+'.png',dpi=600)

np.savetxt(path+ 'Risultati_proiezione//'+'velocities_line_' +str(line)+'_bus_'+str(mezzo)+".txt", data_for_quiver_plot, fmt = '%.0f %.0f %.0f %.0f %s %.3f %.3f %s %.3f %.3f %.3f', header = 'ID1 ID2 n1 n2 t1 x1 y1 t2 x2 y2 v')

plt.show()
