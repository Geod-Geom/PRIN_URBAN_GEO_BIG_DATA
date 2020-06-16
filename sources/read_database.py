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
import osmnx as ox
from collections import Counter
import pdb
import matplotlib.cm as cm

import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg')
print '#1 Backend:',plt.get_backend()
import glob

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


#start = datetime.datetime.now()	
	
path = os.path.dirname(os.path.abspath(__file__))+'//..//' #esco dalla sub directory codice

name_database = path + '\\fcd\\csv_database_formatted_short.db'
#file = path + '\\fcd\\csv_database_formatted.db' #30 milioni di righe

# If not already present, create the directory to store the downloaded data
results_directory = 'Risultati'
if not os.path.exists(path+'\\'+results_directory):
    os.makedirs(path+'\\'+results_directory)





con = lite.connect(name_database)

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
	data = np.asarray(rows, dtype = object)
	rows = None
	

	
	
''' 
		0 - index
		1 - linea
		2 - turno
		3 - date
		4 - mezzo
		5 - lat
		6 - lon
'''
	#sorted_mezzo = data[np.argsort(data[:, 4])]
	#sorted_linea = sorted_mezzo[np.argsort(sorted_mezzo[:, 1])]
	#iso_mezzo[:,3].map(lambda x:  datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S.%f'))

	
linee_histo = find_histogram_class_from_unicode(data[:,1])
val_linee = linee_histo[0]
occ_linee = linee_histo[1]
print  val_linee, occ_linee

print 'Tot linee:', len(val_linee)
for i in xrange(0,len(val_linee)):#ricordarsi di ripartire da 0
	#if i == 2:
	#	sys.exit()
	linea = val_linee[i]#u'36  '
	print linea
	# cerco la linea specifica tra tutte le linee, creando un'apposita maschera
	mask = data[:,1] == linea
	print 'righe per la linea',linea, ':', np.sum(mask)
	
	iso_linea = data[mask]
	if len(iso_linea) != 0:
		mezzo_histo = find_histogram_class_from_int(iso_linea[:,4])
		val_mezzo = mezzo_histo[0][:,0]
		occ_mezzo = mezzo_histo[0][:,1]
		print  val_mezzo, occ_mezzo
		coord_iso_mezzi = []
		print 'Tot mezzi per lnea' , len(val_mezzo)
		
		for j in xrange(0,len(val_mezzo)):
			mezzo = val_mezzo[j]
			print 'mezzo' , mezzo
			
			mask = iso_linea[:,4] == mezzo
			print 'righe per il mezzo',mezzo, ':', np.sum(mask)
			iso_mezzo  = iso_linea[mask]
						
			iso_mezzo[:,3] = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f').date() for date in iso_mezzo[:,3]]
			date_sorted_iso_mezzo = iso_mezzo[np.argsort(iso_mezzo[:, 3])]
			
			
			coord_iso_mezzi.append(date_sorted_iso_mezzo[:,4:])
			
			'''
			# codice per graficare separatamente ogni mezzo di una linea
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
		# ciclo per graficare tutti i mezzi di una linea insieme
		colors = cm.rainbow(np.linspace(0, 1, len(coord_iso_mezzi)))
		
		# bounding box che contiene tutte le posizioni dei mezzi che hanno servito quella linea
		x_inf = np.min(iso_linea[:,6])-0.01
		x_sup = np.max(iso_linea[:,6])+0.01
		y_inf = np.min(iso_linea[:,5])-0.01
		y_sup = np.max(iso_linea[:,5])+0.01 

		G = ox.graph_from_bbox(y_sup, y_inf, x_sup, x_inf, network_type='drive', name = 'prova')

		fig, ax = ox.plot_graph(G, close = False, show=False, axis_off=False)
		for k in xrange(0,len(coord_iso_mezzi)):
			plt.plot(coord_iso_mezzi[k][:,2],coord_iso_mezzi[k][:,1], 'o', color=colors[k], label = 'mezzo '+str(coord_iso_mezzi[k][:,0][0]), markeredgecolor='k')
		plt.xlabel('longitude')
		plt.ylabel('latitude')
		lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
		fig.savefig(path+'//'+results_directory+'//'+'linea_'+linea +'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')
	#plt.show()
	sys.exit()
	


#sys.exit()
print 'Database read in', (datetime.datetime.now() - start).seconds, 'seconds'
plt.show()
