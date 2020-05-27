# -*- coding: utf-8 -*-
'''
    Script name: .py
    Description: the script plots the data contained in the route.shp files
                 on top of the OSM Turin network
    Authors: Roberta Ravanelli   <roberta.ravanelli@uniroma1.it>,
             Geodesy and Geomatics Division (DICEA) @ University of Rome "La Sapienza"
    Python Version: 2.7
    https://gis.stackexchange.com/questions/131716/plot-shapefile-with-matplotlib
    https://stackoverflow.com/questions/24415806/coordinate-of-the-closest-point-on-a-line
    #input: coordinates of srops
    http://all-geo.org/volcan01010/2012/11/change-coordinates-with-pyproj/
    
    '''

import numpy as np
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import sqlite3 as lite
import sys
import os
import pdb
#https://www.datacamp.com/community/tutorials/networkx-python-graph-tutorial
import networkx as nx
import pandas as pd
import shapefile as shp
from datetime import datetime
from datetime import timedelta
#from operator import itemgetter
from matplotlib import rc,rcParams
import pdb

plt.close('all')
rc('font', weight='bold')
# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import cv2

from pyproj import Proj, transform


from contextlib import contextmanager
@contextmanager
def nested_break():
    class NestedBreakException(Exception):
        pass
    try:
        yield NestedBreakException
    except NestedBreakException:
        pass

def find_histogram_class_from_int(data_int):
    from collections import Counter
    valoriEfrequenze=Counter(data_int)#serve per la moda: genera un array con due colonne (il valore dell'array e la relativa frequenza) e n classi (righe)
    n=len(valoriEfrequenze.most_common())# mi dice il numero delle classi dell'istogramma
    valori=np.zeros(n)
    occorrenze=np.zeros(n)
    for i in xrange(0, n):#così ordina le frequenze in senso crescente:bins must increase monotonically
         occorrenze[i]=valoriEfrequenze.most_common()[n-1-i][1]#primo indice righe, secondo colonne:la prima colonna è il valore, la seconda è la frequenza
         valori[i]=valoriEfrequenze.most_common()[n-1-i][0]
    moda=valori[n-1]
    return (np.column_stack((valori, occorrenze))).astype(int), moda,n

def isBetween(P1, P2, P, epsilon = 0.000000001):
    #https://stackoverflow.com/questions/328107/how-can-you-determine-a-point-is-between-two-other-points-on-a-line-segment
    crossproduct = (P[1] - P1[1]) * (P2[0] - P1[0]) - (P[0] - P1[0]) * (P2[1] - P1[1])
    #if abs(crossproduct) > 0.000000001 : return False   # (or != 0 if using integers)
    if abs(crossproduct) >  epsilon : return False
    
    dotproduct = (P[0] - P1[0]) * (P2[0] - P1[0]) + (P[1] - P1[1])*(P2[1] - P1[1])
    if dotproduct < 0 : return False

    squaredlengthba = (P2[0]- P1[0])*(P2[0] - P1[0]) + (P2[1] - P1[1])*(P2[1] - P1[1])
    if dotproduct > squaredlengthba: return False

    return True

def order_points(pts):
    #https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

def read_db (filename, tablename='table'):
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

path = os.path.dirname(os.path.abspath(__file__))+'//..//'

# Le colonne di FCD_no_outlier sono tutte stringhe: devo convertirle in numeri
#FCD_no_outlier = np.genfromtxt(path+"\\databases\\velocities_nuove\\shape\\shape_velocities_line_39.csv", delimiter = ';', dtype = object, skip_header =1)
FCD_no_outlier = np.genfromtxt(path+"//databases//velocities_nuove//shape//shape_velocities_line_39.csv", delimiter = ';', dtype = object, skip_header =1)
#print  FCD_no_outlier[0]
FCD_no_outlier[:,7] = FCD_no_outlier[:,7].astype(np.float)
FCD_no_outlier[:,8] = FCD_no_outlier[:,8].astype(np.float) 
FCD_no_outlier[:,-1] = FCD_no_outlier[:,-1].astype(np.int)
FCD_no_outlier[:,0] = FCD_no_outlier[:,0].astype(np.int)
FCD_no_outlier[:,1] = [datetime.strptime(date, '%Y/%m/%d %H:%M:%S.%f') for date in FCD_no_outlier[:,1].astype(str)]  
#print  FCD_no_outlier[0]


mezzo_histo = find_histogram_class_from_int(FCD_no_outlier[:,0])
val_mezzo = mezzo_histo[0][:,0]
occ_mezzo = mezzo_histo[0][:,1]
print val_mezzo
print occ_mezzo
mezzo = 8265 #363 funziona, 2584 funziona tranne 2 punti, 8265
mask = FCD_no_outlier[:,0] == mezzo



print 'righe per il mezzo',mezzo, ':', np.sum(mask)
iso_mezzo  = FCD_no_outlier[mask]#[:10,]

#u, indices = np.unique(lista_punti_reprojected[:,2], return_index=True)
#lista_punti_reprojected[indices]


# ultima colonna: flag outlier
# rimuovo quelli con flag = 1 (outlier)
print FCD_no_outlier
print len(FCD_no_outlier)
FCD_no_outlier = FCD_no_outlier[ FCD_no_outlier[:,-1] == 0]
print len(FCD_no_outlier)




#lon e lat erano invertite
#FCD = FCD_no_outlier [:,7:9].astype(np.float)
#FCD = np.zeros((len(FCD_no_outlier),3), dtype = object)old
FCD = np.zeros((len(iso_mezzo),3), dtype = object)
FCD[:,0] = iso_mezzo[:,8].astype(np.float)# lon, E
FCD[:,1] = iso_mezzo[:,7].astype(np.float)# lat, N
FCD[:,2] = iso_mezzo[:,1]
#FCD = FCD[60:]
# conversion to web mercator
# https://gis.stackexchange.com/questions/247871/convert-gps-coordinates-to-web-mercator-epsg3857-using-python-pyproj
for index in range (len(FCD)):
    E, N = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), FCD[index, 0], FCD[index, 1])
    FCD[index, 0], FCD[index, 1] = E, N

FCD[:,:2] = FCD[:,:2].astype(np.float)




data_stops = read_db(path+'databases//routes_databases//routes_database_39.db','route_table')


# conversion to web mercator
# https://gis.stackexchange.com/questions/247871/convert-gps-coordinates-to-web-mercator-epsg3857-using-python-pyproj
for index in range (len(data_stops)):
    data_stops[index, 0], data_stops[index, 1] = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), data_stops[index, 0], data_stops[index, 1])


data_stops.astype(np.float)
#sys.exit()


stops_polyline = LineString(data_stops)


# polilinea e data stops sono uguali
polilinea = np.array(list(stops_polyline.coords))
'''plt.figure('Linea 1')
plt.plot(polilinea[:,0],polilinea[:,1], 'bo--')
cont=0
for  i in xrange(0, len(polilinea)):
    label = '%d' % (cont)
    plt.text(polilinea[i,0]+0.0001, polilinea[i,1]+0.0001, label, color='b',fontsize=9)
    plt.margins(0.1)
    cont=cont+1'''



# Il gps acquisce con un errore: le coordinate sono fuori dal vero tracciato
coord_outside_network = FCD

'''for j in range (len(coord_outside_network)-1):
    print format(100 *j/(len(coord_outside_network)-1),'.2f')
    plt.quiver( coord_outside_network[j,0], # start x
                coord_outside_network[j,1], # start y
                coord_outside_network[j+1,0] - coord_outside_network[j,0], # delta x 
                coord_outside_network[j+1,1] - coord_outside_network[j,1], # delta y
                angles='xy', # ‘xy’: arrows point from (x,y) to (x+dx, y+dy). Use this for plotting a gradient field, for example.
                scale=1, # più è grande, + le frecce sono corte Number of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer. The arrow length unit is given by the scale_units paramete
                scale_units='xy', # usando le scale units, non è più necessario alterare la scala #to plot vectors in the x-y plane, with u and v having the same units as x and y, use angles='xy', scale_units='xy', scale=1.
                color= 'black',# color = velocities
                zorder = 5, #più è alto, più il plot è in primo piano
                edgecolor='k', # colore bordo freccia
                linewidth=.4,
                alpha=0.6) '''

points_outside_network = []

# -1 perchè per convenzione associo i punti al nodo iniziale
# quindi all' ultimo nodo non posso associare nessun punto
lista_proj_points_for_ramo = [ [] for _ in range(len(polilinea) -1)]

# questo ciclo associa ogni punto acquisito dal GPS
# a un ramo (segmento tra 2 fermate consecutive) del tracciato
# calcolando la distanza piu  
stop_i_old = 100000000000000000000
t_old = coord_outside_network[0, 2]

# ciclo su tutti i punti dei FCD
for index in xrange(0, len(coord_outside_network)):

    print format(100 *index/(len(coord_outside_network)-1),'.2f'), '% '
    cont_true = 0
    # mi serve l'oggetto point
    point_outside_network = Point(coord_outside_network[index, 0], coord_outside_network[index, 1])

    # Now combine with interpolated point on line
    point_inside_network = stops_polyline.interpolate(stops_polyline.project(point_outside_network))
    print
    #print point_inside_network
    #https://stackoverflow.com/questions/24415806/coordinate-of-the-closest-point-on-a-line
    xy_P_projected = np.array(list(point_inside_network.coords)[0])
    boh = np.array(list(point_outside_network.coords)[0])
    print "P",index, ' ', point_inside_network, ' ', xy_P_projected, ' ', coord_outside_network[index, 2] #t1
    #pdb.set_trace()
    EPS = 0.000001
    polilinea_appoggio = polilinea.copy()
    # ciclo su tutta 
    for i in range(len(polilinea)-1):
            # questo if serve a capire su quale ramo siamo (lo sappiamo già visivamente perchè il punto è già proiettato
            # e probabilmente è un'informazione che la funzione dovrebbe già avere, ma dobbiamo capirlo noi)
            # se la seguente condizione non è verificata, c'è un problema: il punto proiettato non appartiene a nessun ramo
            # il problema è che questo controllo andrebbe fatto prima della proiezione!!!
            if isBetween(polilinea[i], polilinea[i+1], xy_P_projected, epsilon =EPS) == True:# qui forse va polilinea appoggio?!?
                print stop_i_old
                #i = i + 1
                selection = i
                #stop_i_old = i

                # può succedere che selection > (stop_i_old - 20)
                while selection > (stop_i_old + 20): #or selection < (stop_i_old):
                    #cont_true = cont_true+1
                    #print stop_i_old
                    print 'ciao'
                    polilinea_appoggio = np.delete(polilinea_appoggio, (selection, selection+1), axis=0)
                    
                    if len(polilinea_appoggio)==1:
                        print 'ops'
                        pdb.set_trace()
                        break # non riesco ad assegnare il punto

                    stops_polyline_appoggio = LineString(polilinea_appoggio[:,:2])

                    # non sono soddisfatta (il ramo prescelto è troppo distante dal ramo precedente) e riproietto
                    point_inside_network = stops_polyline_appoggio.interpolate(stops_polyline_appoggio.project(point_outside_network))
                    xy_P_projected = np.array(list(point_inside_network.coords)[0])
                    for ii in range(len(polilinea_appoggio)-1):
                          if isBetween(polilinea_appoggio[ii], polilinea_appoggio[ii+1],xy_P_projected, epsilon = EPS) == True:# and ii >= stop_i_old:
                                cont_true = cont_true+1
                                #if ii < stop_i_old:
                                #    ii =  ii + 1
                                selection = ii
                                #stop_i_old = ii
                                #break
                                '''if selection < stop_i_old :
                                    print '0ps2'
                                    pdb.set_trace()'''
                stop_i_old = selection
                cont_true = cont_true + 1
                print stop_i_old
                print "The point belongs to the segment ", selection, selection+1
                lista_proj_points_for_ramo[selection].append([xy_P_projected[0], xy_P_projected[1], coord_outside_network[index,2]])

    '''plt.plot(coord_outside_network[index,0], coord_outside_network[index,1], 'ko')
    plt.text(coord_outside_network[index,0], coord_outside_network[index,1], str(index), color='b',fontsize=9)

    plt.plot(xy_P_projected[0], xy_P_projected[1], 'ro')
    plt.text(xy_P_projected[0], xy_P_projected[1], str(index), color='r',fontsize=9)
    print 'cont true', cont_true
    #pdb.set_trace()
    plt.grid()'''

#sys.exit()
#pdb.set_trace()
#print lista_proj_points_for_ramo

# Aggiungo ad ogni singola fermata anche un dato di tempo fittizio (NaT)
# per questioni di simmetria con gli FCD
#polilinea = np.hstack((polilinea,np.zeros((polilinea.shape[0],1))*pd.NaT))
polilinea = np.hstack((polilinea,np.zeros((polilinea.shape[0],1))*np.nan))

final_path = []
for k in range( len(polilinea)-1):

    #print polilinea[k], 
    final_path.append(polilinea[k])
    for boh in range(len(lista_proj_points_for_ramo[k])):
        #print lista_proj_points_for_ramo[k][boh],
        final_path.append(lista_proj_points_for_ramo[k][boh])



final_path = np.array(final_path, dtype = object)

''''for j in range (1,len(final_path)-1):
    plt.quiver( final_path[j,0], # start x
                final_path[j,1], # start y
                final_path[j+1,0] - final_path[j,0], # delta x 
                final_path[j+1,1] - final_path[j,1], # delta y
                angles='xy', # ‘xy’: arrows point from (x,y) to (x+dx, y+dy). Use this for plotting a gradient field, for example.
                scale=1, # più è grande, + le frecce sono corte Number of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer. The arrow length unit is given by the scale_units paramete
                scale_units='xy', # usando le scale units, non è più necessario alterare la scala #to plot vectors in the x-y plane, with u and v having the same units as x and y, use angles='xy', scale_units='xy', scale=1.
                color= 'r',# color = velocities
                zorder = 5, #più è alto, più il plot è in primo piano
                edgecolor='black', # colore bordo freccia
                linewidth=.4,
                alpha=0.8) '''

                


lista_punti_reprojected = []
cont_punti = 0
print 'ID x y date hour start_tree end_tree'
for k in xrange(len (lista_proj_points_for_ramo)):
    
    #pdb.set_trace()
    if len(lista_proj_points_for_ramo[k]) != 0:
        #https://stackoverflow.com/questions/20183069/how-to-sort-multidimensional-array-by-column
        # ordino i punti all'interno di ogni ramo sulla base del tempo di acquisizione
        #pdb.set_trace()
        #lista_proj_points_for_ramo[k] = sorted(lista_proj_points_for_ramo[k], key=lambda x: x[2])
        lista_proj_points_for_ramo[k].sort(key=lambda x: x[2])
        lista_proj_points_for_ramo[k] = np.array(lista_proj_points_for_ramo[k])
        #pdb.set_trace()
        for kkk in xrange(len (lista_proj_points_for_ramo[k])):
            lista_punti_reprojected.append(np.array([lista_proj_points_for_ramo[k][kkk,0], lista_proj_points_for_ramo[k][kkk,1], lista_proj_points_for_ramo[k][kkk,2], k, k+1]))
            #plt.plot(lista_proj_points_for_ramo[k][kkk,0], lista_proj_points_for_ramo[k][kkk,1], 'ks')# markersize=12)
            #pdb.set_trace()
            
            print cont_punti, lista_proj_points_for_ramo[k][kkk,0], lista_proj_points_for_ramo[k][kkk,1], lista_proj_points_for_ramo[k][kkk,2], k, k+1
            #plt.grid()
            cont_punti = cont_punti +1
lista_punti_reprojected = np.array(lista_punti_reprojected)

# la prima condizione verifica che i tempi siano monoliticamente crescenti
# https://scipython.com/book/chapter-4-the-core-python-language-ii/questions/determining-if-an-array-is-monotonically-increasing/
print 'check ',np.all(np.diff(lista_punti_reprojected[:,2]) > timedelta(seconds=0)), len(FCD) == len(lista_punti_reprojected), np.all(np.diff(lista_punti_reprojected[:,3]) >= 0)
#pdb.set_trace()

# rimuovo i doppioni (non ho capito perchè ci sono)
# il procedimento utilizzato per proiettare i punti crea dei doppioni: stesso tempo, diverso ramo
# non possono esserci due punti con lo stesso tempo di acquisizione
u, indices = np.unique(lista_punti_reprojected[:,2], return_index=True)
lista_punti_reprojected = lista_punti_reprojected[indices]
print 'check ', np.all(np.diff(lista_punti_reprojected[:,2]) > timedelta(seconds=0)), len(FCD) == len(lista_punti_reprojected), np.all(np.diff(lista_punti_reprojected[:,3]) >= 0)

pdb.set_trace()
ramo_histo = find_histogram_class_from_int(lista_punti_reprojected[:,3])
val_ramo = ramo_histo[0][:,0]
occ_ramo = ramo_histo[0][:,1]
print val_ramo 
print occ_ramo

for ggg in range (len(val_ramo)):
    ramo = val_ramo[ggg]
    mask_r = lista_punti_reprojected[:,3] == ramo
    print 'righe per il ramo',ramo, ':', np.sum(mask_r)
    iso_ramo  = lista_punti_reprojected[mask_r]
    #https://stackoverflow.com/questions/20183069/how-to-sort-multidimensional-array-by-column
    iso_ramo = sorted(iso_ramo,key=lambda x: x[2])

print "\nRisultati"
for index in range(len(lista_punti_reprojected)):
    print str(index).zfill(2) , format(lista_punti_reprojected[index,0],'.3f'), format(lista_punti_reprojected[index,1],'.3f'), lista_punti_reprojected[index,2].strftime('%Y/%m/%d %H:%M:%S'), lista_punti_reprojected[index,3], lista_punti_reprojected[index,4]


plt.figure('boh '+str(mezzo))
plt.plot(polilinea[:,0],polilinea[:,1], 'ko--')
plt.plot(coord_outside_network[:,0], coord_outside_network[:,1], 'bo')
plt.xlabel('E [m]', fontweight='bold')#)
plt.ylabel('N [m]', fontweight='bold')#, fontsize=14, fontweight='bold')


for zz in range(len(coord_outside_network)):
    plt.text(coord_outside_network[zz,0]-0.00001, coord_outside_network[zz,1]-0.00001, str(zz) , color='b',fontsize=9 )
for zz in range(len(polilinea)):
    plt.text(polilinea[zz,0]-0.00001, polilinea[zz,1]-0.00001, str(zz) , color='k',fontsize=9 )


#plt.plot(lista_punti_reprojected[:,0], lista_punti_reprojected[:,1], 'ro')
for kkk in range(len(lista_punti_reprojected)):
    plt.plot(lista_punti_reprojected[kkk,0], lista_punti_reprojected[kkk,1], 'ro')
    #plt.text(lista_punti_reprojected[kkk,0],+0.0001, lista_punti_reprojected[kkk,0],+0.0001, lista_punti_reprojected[kkk,2].strftime('%Y/%m/%d %H:%M:%S') , color='b',fontsize=9, )
    #label = '%d' % (kkk)
    #plt.text(lista_punti_reprojected[kkk,0]-0.00005, lista_punti_reprojected[kkk,1]-0.00005, label , color='r',fontsize=9 )
    #plt.text(lista_punti_reprojected[kkk,0]+0.00005, lista_punti_reprojected[kkk,1]+0.00005, str(kkk) + '\n'+lista_punti_reprojected[kkk,2].strftime('%y/%m/%d\n%H:%M:%S') , color='r',fontsize=9 )
    plt.text(lista_punti_reprojected[kkk,0]+0.00005, lista_punti_reprojected[kkk,1]+0.00005, str(kkk) + '   '+lista_punti_reprojected[kkk,2].strftime('%H:%M:%S') , color='r',fontsize=9 )
   
    plt.margins(0.1)
    
    '''for j in range (1,len(lista_punti_reprojected)-1):
    plt.quiver( lista_punti_reprojected[j,0], # start x
                lista_punti_reprojected[j,1], # start y
                lista_punti_reprojected[j+1,0] - lista_punti_reprojected[j,0], # delta x 
                lista_punti_reprojected[j+1,1] - lista_punti_reprojected[j,1], # delta y
                angles='xy', # ‘xy’: arrows point from (x,y) to (x+dx, y+dy). Use this for plotting a gradient field, for example.
                scale=1, # più è grande, + le frecce sono corte Number of data units per arrow length unit, e.g., m/s per plot width; a smaller scale parameter makes the arrow longer. The arrow length unit is given by the scale_units paramete
                scale_units='xy', # usando le scale units, non è più necessario alterare la scala #to plot vectors in the x-y plane, with u and v having the same units as x and y, use angles='xy', scale_units='xy', scale=1.
                color= 'r',# color = velocities
                zorder = 5, #più è alto, più il plot è in primo piano
                edgecolor='black', # colore bordo freccia
                linewidth=.4,
                alpha=0.8) 
'''
plt.grid()
plt.axis('equal')
plt.tight_layout()

'''
###### Figura per articolo #####
plt.figure('boh2 '+str(mezzo))
plt.plot(polilinea[:,0],polilinea[:,1], 'ko--',  markersize=8)
plt.plot(coord_outside_network[:,0], coord_outside_network[:,1], 'bo', markersize=8)
#plt.plot(lista_punti_reprojected[:,0], lista_punti_reprojected[:,1], 'ro',  markersize=8)
plt.xlabel('E [m]', fontweight='bold')#)
plt.ylabel('N [m]', fontweight='bold')#, fontsize=14, fontweight='bold')
plt.grid()
plt.axis('equal')
plt.tight_layout()
#plt.savefig('C://Users//Roberta//Desktop//figurearticolo//figure1.png')
'''

fig, ax1 = plt.subplots()
#t = np.arange(0.01, 10.0, 0.01)
#s1 = np.exp(t)
ax1.plot(lista_punti_reprojected[:,2], 'bo--')
ax1.set_xlabel('index')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('time', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(lista_punti_reprojected[:,3], 'ro--')
ax2.set_ylabel('primo nodo ramo', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.grid()
'''plt.figure()
plt.plot(lista_punti_reprojected[:,2])
plt.plot(lista_punti_reprojected[:,3])'''

plt.show()


