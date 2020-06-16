# -*- coding: utf-8 -*-
'''
    #https://gis.stackexchange.com/questions/46657/how-to-interpolate-polyline-or-line-from-points
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
from scipy.spatial import distance as dist
from pyproj import Proj, transform

plt.close('all')
rc('font', weight='bold')

def dist_point_from_segment(x1,y1, x2,y2, x3,y3): # x3,y3 is the point, x1, y1 is the start of the segment, x2, y2 is the end point of the segment
    #https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = np.sqrt(dx*dx + dy*dy)

    return dist


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

def get_tree_for_point(point_inside_network, polilinea, eps):
    for i in range(len(polilinea)-1):
            # questo if serve a capire su quale ramo siamo (lo sappiamo già visivamente perchè il punto è già proiettato
            # e probabilmente è un'informazione che la funzione dovrebbe già avere, ma dobbiamo capirlo noi)
            # se la seguente condizione non è verificata, c'è un problema: il punto proiettato non appartiene a nessun ramo
            #selection = i
            if isBetween(polilinea[i], polilinea[i+1], point_inside_network, epsilon =eps) == True:# qui forse va polilinea appoggio?!?
                #print "The point belongs to the segment ", i, i+1
                # implicitamente sto assegnando il punto al primo ramo che verifica la condizione
                # forse e sottolineo il forse, potrebbero esserci casi in cui ci sono più rami per lo stesso punto
                return i, i+1
    return np.nan, np.nan

def roby (coord_outside_network, data_stops, star_tree =None):

    #coord_outside_network 0->x, 1-> y, 2->timestamps
    # converto data_stops (array di numpy) in un oggetto di shapely
    stops_polyline = LineString(data_stops)


    # Il gps acquisce con un errore: le coordinate sono fuori dal vero tracciato

    # -1 perchè per convenzione associo i punti al nodo iniziale
    # quindi all' ultimo nodo non posso associare nessun punto
    lista_proj_points_for_ramo = [ [] for _ in range(len(data_stops) -1)]

    # questo ciclo associa ogni punto acquisito dal GPS
    # a un ramo (segmento tra 2 fermate consecutive) del tracciato
    # sulla base della distanza più breve dalla polilinea
    # trova il segmento della polilinea più vicino al punto
    # problema con il rumore del GPS

    EPS = 0.000001

    # ciclo su tutti i punti dei FCD
    # per proiettarli sul ramo più vicino del network
    # che non è detto sia quello giusto
    
    
    print "Roby"
    # storage for results
    
    #lista_punti_reprojected = np.zeros((len(coord_outside_network),8))
    #lista_punti_reprojected[:,3] = lista_punti_reprojected[:,3].astype(datetime)
    lista_punti_reprojected = []
    point_outside_network = Point(coord_outside_network[0, 0], coord_outside_network[0, 1])
    # Now combine with interpolated point on line: proietto tutti i punti sulla polilinea
    point_inside_network = stops_polyline.interpolate(stops_polyline.project(point_outside_network))
    # distanza del punto dal nodo (ramo) più vicino
    d_outside_network = point_outside_network.distance(stops_polyline)
    xy_P_projected = np.array(list(point_inside_network.coords)[0])
    nodo_seed, nodo_piu_uno = get_tree_for_point(xy_P_projected, data_stops, EPS)
    
    tree_previous_point = nodo_seed# seed
    point_tree = - 10
    # controllo su tutti i punti le distanze da tutti i nodi
    for index in range(0,len(coord_outside_network)):
        point_distances = np.zeros((len (data_stops) - 1,3))
        #polilinea_appoggio = data_stops.copy()
        print '\nPoint ', index

        for node in range( len (data_stops) - 1):
            x1 = data_stops[node,0]
            y1 = data_stops[node,1]
            x2 = data_stops[node + 1,0]
            y2 = data_stops[node +1,1]
            d_from_tree =dist_point_from_segment(x1,y1, x2,y2, coord_outside_network[index,0],coord_outside_network[index,1])

            point_distances[node,0] = node
            point_distances[node,1] = node +1
            point_distances[node,2] = d_from_tree

        point_distances = point_distances[point_distances[:,2].argsort()]
        point_distances = point_distances[:11,:]
        print 'ramo + vicino', int(point_distances[0,0]), int(point_distances[0,1])
        print point_distances
        
        if point_distances[0,2] > 400:
            print 'probabile outlier'# non lo considero
            point_tree = -1
        elif point_distances[0,0] > tree_previous_point + 20:
            print 'problema saltone'
            for i in range (1, len(point_distances)):
                if point_distances[i,0] < point_distances[0,0]-50:
                    print 'real node' , point_distances[i,0]
                    point_tree = point_distances[i,0]
                    break
                else:# la condizione in casi particolari potrebbe non verificarsi mai
                    point_tree = point_distances[1,0]
        elif point_distances[0,0] < tree_previous_point:
            print 'problema derivata negativa'
            for i in range (1, len(point_distances)):
                print i
                if point_distances[i,0] > point_distances[0,0]+50:
                    potential_previous_point = point_distances[i,0]
                    break 
                else:# la condizione in casi particolari potrebbe non verificarsi mai
                    potential_previous_point = point_distances[1,0]
                    
            if potential_previous_point < tree_previous_point:
                    print 'new lap'# devo dare priorità al nodo con ID piccolo
                    for i in range (1, len(point_distances)):
                        if point_distances[i,0] < 5:# point_distances[0,0]+50:
                            print 'real node' ,point_distances[i,0]
                            point_tree = point_distances[i,0]
                            break
            else:
                    point_tree = potential_previous_point
                    print 'real node ', potential_previous_point

        else:
            # tutto ok, il ramo più vicino è quello giusto 
            point_tree = point_distances[0,0]
            print 'real node ', int(point_distances[0,0])

        print ' REAL NODE ',point_tree

        if point_tree != -1:
            tree_previous_point = point_tree
            stops_polyline_appoggio = LineString(data_stops[int(point_tree):int(point_tree)+2])
            point_outside_network = Point(coord_outside_network[index, 0], coord_outside_network[index, 1])
            # Now combine with interpolated point on line: proietto tutti i punti sulla polilinea
            point_inside_network = stops_polyline_appoggio.interpolate(stops_polyline_appoggio.project(point_outside_network))
            # distanza del punto dal nodo (ramo) più vicino
            #d_outside_network = point_outside_network.distance(stops_polyline_appoggio)
            xy_P_projected = np.array(list(point_inside_network.coords)[0])
            nodo, nodo_piu_uno = get_tree_for_point(xy_P_projected, data_stops, EPS)
            print 'check', nodo
            #pdb.set_trace()
            lista_punti_reprojected.append(np.array([xy_P_projected[0],xy_P_projected[1],coord_outside_network[index,2],point_tree,point_tree+1, coord_outside_network[index,0],coord_outside_network[index,1]]))
            '''
            lista_punti_reprojected[index, 0] = index
            lista_punti_reprojected[index, 1] = xy_P_projected[0]
            lista_punti_reprojected[index, 2] = xy_P_projected[1]
            lista_punti_reprojected[index, 3] = coord_outside_network[index,2]
            
            lista_punti_reprojected[index, 4] = point_tree
            lista_punti_reprojected[index, 5] = point_tree + 1
            lista_punti_reprojected[index, 6] = coord_outside_network[index,0]
            lista_punti_reprojected[index, 7] = coord_outside_network[index,1]
            '''
            '''
            if (index > 63):
                plt.figure('Point '+str(index))
                plt.plot(data_stops[:,0],data_stops[:,1], 'ko--')
                for zz in range(len(data_stops)):
                    plt.text(data_stops[zz,0]-0.00001, data_stops[zz,1]-0.00001, str(zz) , color='k',fontsize=12 )
                plt.plot(coord_outside_network[index,0], coord_outside_network[index,1], 'bo', linewidth=2,  markersize=8)
                plt.plot(xy_P_projected[0], xy_P_projected[1], 'ro', linewidth=2,  markersize=8)
                plt.xlabel('E [m]', fontweight='bold')#)
                plt.ylabel('N [m]', fontweight='bold')#, fontsize=14, fontweight='bold')
                plt.ticklabel_format(useOffset=False, style='plain')
                plt.grid()
                plt.show()
                pdb.set_trace()
            '''
            
        else:
            lista_punti_reprojected.append(np.array([np.nan,np.nan,coord_outside_network[index,2],point_tree, np.nan, coord_outside_network[index,0],coord_outside_network[index,1]]))
        #pdb.set_trace()
      
        

        
    lista_punti_reprojected = np.array(lista_punti_reprojected)
    pdb.set_trace()
    print 'check ', np.all(np.diff(lista_punti_reprojected[:,2]) > timedelta(seconds=0)), len(FCD) == len(lista_punti_reprojected), np.all(np.diff(lista_punti_reprojected[:,3]) >= 0)   



    return lista_punti_reprojected

def get_distance_from_netwowork(coord_outside_network, data_stops):
    stops_polyline = LineString(data_stops)
    distances = np.zeros(len(coord_outside_network))
    for index in xrange(0, len(coord_outside_network)):

        # mi serve l'oggetto point
        point_outside_network = Point(coord_outside_network[index, 0], coord_outside_network[index, 1])
        d_outside_network = point_outside_network.distance(stops_polyline)
        
        #pdb.set_trace()
        #print point_inside_network
        #https://stackoverflow.com/questions/24415806/coordinate-of-the-closest-point-on-a-line
        print "P",index, ' ', d_outside_network
        distances[index] = d_outside_network
    return distances

def plot_results(lista_punti_reprojected, data_stops, line, mezzo):

    plt.figure('Line ' +str(line)+' bus '+str(mezzo))
    plt.plot(data_stops[:,0],data_stops[:,1], 'ko--')
    plt.plot(lista_punti_reprojected[:,5], lista_punti_reprojected[:,6], 'bo', linewidth=2,  markersize=8)# originale FCD coordinates
    plt.xlabel('E [m]', fontweight='bold')#)
    plt.ylabel('N [m]', fontweight='bold')#, fontsize=14, fontweight='bold')
    plt.ticklabel_format(useOffset=False, style='plain')

    for zz in range(len(lista_punti_reprojected)):
        plt.text(lista_punti_reprojected[zz,5]-0.00001, lista_punti_reprojected[zz,6]-0.00001, str(zz) , color='b',fontsize=12 )
    for zz in range(len(data_stops)):
        plt.text(data_stops[zz,0]-0.00001, data_stops[zz,1]-0.00001, str(zz) , color='k',fontsize=12 )


    #plt.plot(lista_punti_reprojected[:,0], lista_punti_reprojected[:,1], 'ro')
    for kkk in range(len(lista_punti_reprojected)):
        plt.plot(lista_punti_reprojected[kkk,0], lista_punti_reprojected[kkk,1], 'ro', markersize=8, )
        #plt.text(lista_punti_reprojected[kkk,0],+0.0001, lista_punti_reprojected[kkk,0],+0.0001, lista_punti_reprojected[kkk,2].strftime('%Y/%m/%d %H:%M:%S') , color='b',fontsize=9, )
        #label = '%d' % (kkk)
        #plt.text(lista_punti_reprojected[kkk,0]-0.00005, lista_punti_reprojected[kkk,1]-0.00005, label , color='r',fontsize=9 )
        #plt.text(lista_punti_reprojected[kkk,0]+0.00005, lista_punti_reprojected[kkk,1]+0.00005, str(kkk) + '\n'+lista_punti_reprojected[kkk,2].strftime('%y/%m/%d\n%H:%M:%S') , color='r',fontsize=9 )
        plt.text(lista_punti_reprojected[kkk,0]+0.00006, lista_punti_reprojected[kkk,1]+0.00006, str(kkk) + '   '+lista_punti_reprojected[kkk,2].strftime('%H:%M:%S') , color='r',fontsize=12 )
        plt.margins(0.1)
    plt.grid()
    plt.axis('equal')


    # 2nd plot
    fig, ax1 = plt.subplots()
    fig.canvas.set_window_title('Temporal trend line ' +str(line)+' bus '+str(mezzo))
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


###################################################################################################################
path = os.path.dirname(os.path.abspath(__file__))+'//..//'
line = 39
# Le colonne di FCD_no_outlier sono tutte stringhe: devo convertirle in numeri
#FCD_no_outlier = np.genfromtxt(path+"\\databases\\velocities_nuove\\shape\\shape_velocities_line_39.csv", delimiter = ';', dtype = object, skip_header =1)
FCD_no_outlier = np.genfromtxt(path+"//databases//velocities_nuove//shape//shape_velocities_line_"+str(line)+".csv", delimiter = ';', dtype = object, skip_header =1)
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
mezzo = 309#2581 ok, ma verificato fino al punto 59, 363 funziona, 2584 funziona tranne 2 punti, 8265 funziona a parte pochi punti, 309 andamento ciclico sbagliato perchè non riconosce il nuovo giro
mask = FCD_no_outlier[:,0] == mezzo


print 'righe per il mezzo',mezzo, ':', np.sum(mask)
iso_mezzo  = FCD_no_outlier[mask]#[:10,]

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

if mezzo == 2581:
    FCD = FCD[12:]#FCD[12:]
# conversion to web mercator
# https://gis.stackexchange.com/questions/247871/convert-gps-coordinates-to-web-mercator-epsg3857-using-python-pyproj
for index in range (len(FCD)):
    E, N = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), FCD[index, 0], FCD[index, 1])
    FCD[index, 0], FCD[index, 1] = E, N

FCD[:,:2] = FCD[:,:2].astype(np.float)


data_stops = read_db(path+'databases//routes_databases//routes_database_'+str(line)+'.db','route_table')


# conversion to web mercator coordinates (E,N from phi and lambda)
# https://gis.stackexchange.com/questions/247871/convert-gps-coordinates-to-web-mercator-epsg3857-using-python-pyproj
for index in range (len(data_stops)):
    data_stops[index, 0], data_stops[index, 1] = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), data_stops[index, 0], data_stops[index, 1])


distances =get_distance_from_netwowork(FCD, data_stops)

#pdb.set_trace()

#FCD = FCD[distances  < np.median(distances)*4]

# proietto gli FCD sul tracciato della linea
results = roby (FCD, data_stops)


with open(path+"Risultati_proiezione/"+'linea_'+str(line)+'_mezzo_'+str(mezzo)+".txt",'w') as f:
    f.write("Point_ID\tE_proj\tN_proj\tDay\tHour\tstart_node\tend_node\tE_or\tN_or\n")
    print "\nRisultati FINALI mezzo", mezzo
    for index in range(len(results)):
        print str(index).zfill(3) , format(results[index,0],'.3f'), format(results[index,1],'.3f'), results[index,2].strftime('%Y/%m/%d %H:%M:%S'), results[index,3], results[index,4], format(results[index,5],'.3f'), format(results[index,6],'.3f')#, format(results[index,7],'.3f'), format(results[index,8],'.3f')  
        f.write(str(index).zfill(3) +'\t'+ format(results[index,0],'.3f') +'\t'+ format(results[index,1],'.3f') +'\t'+ results[index,2].strftime('%Y/%m/%d\t%H:%M:%S') +'\t'+ format(results[index,3],'.0f') +'\t'+ format(results[index,4],'.0f')+'\t'+format(results[index,5],'.3f')+'\t'+ format(results[index,6],'.3f')+'\n')

#lista_punti_reprojected = lista_punti_reprojected[~np.isnan(lista_punti_reprojected[:,1].astype(float))]
        
plot_results(results, data_stops, str(line) + ' final', mezzo)

#np.all(np.diff(results[:61,3]) >= 0)

#plt.figure('histo distances outside network')
#n_t,bins_t,patches=plt.hist(lista_punti_reprojected[:,7],facecolor='c',histtype='bar',ec='black')
plt.show()


##################################à

