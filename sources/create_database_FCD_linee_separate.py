# -*- coding: utf-8 -*-
# http://pythondata.com/working-large-csv-files-python/
# https://plot.ly/python/big-data-analytics-with-pandas-and-sqlite/
# https://stackoverflow.com/questions/43607206/issue-with-saving-undefined-table-columns-with-pandas-to-sql
# http://blog.districtdatalabs.com/simple-csv-data-wrangling-with-python

import os
import pandas as pd
from sqlalchemy import create_engine # database connection
import sys
import datetime as dt
from collections import Counter
import numpy as np


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


path = os.path.dirname(os.path.abspath(__file__))+'//..//' #esco dalla sub directory codice
file = path + '\\fcd\\fcd.csv'




# To inspect the database, we print the first 10 rows
# in order to understand the structure of the csv file and 
# make sure the data is formatted in a way that makes sense for the subsequent analysis
# since there is no header, we rename the columns 
df0 = pd.read_csv(file, nrows=10, delimiter =';', skipinitialspace= True, lineterminator= '\n', names =['index', 'linea', 'turno(?)', 'date' , 'mezzo', 'lat', 'lon'])
#print df0.columns.values
print df0

# Initializes database with filename csv_database_formatted.db in current directory
# ogni volta va cancellato, altrimenti si sdoppiano le righe
#csv_database = create_engine('sqlite:///..///csv_database_formatted.db')
start = dt.datetime.now()
chunksize = 100000
cont = 0
index_start = 1
print '\nInizio creazione database'

# The for loop read a chunk of data from the CSV file, 
# replaces the commas with points from in  of column names, 
# then stores the chunk into the sqllite database (df.to_sql(…))
# one separated database for each linea
for df in pd.read_csv(file, chunksize=chunksize, iterator=True, delimiter =';', skipinitialspace= True, lineterminator= '\n', names =['index', 'linea', 'turno(?)', 'date' , 'mezzo', 'lat', 'lon']):
        print cont
        # Remove spaces from columns
        df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})

        # Replace the comma in the coordinate with a point 
        # for every (!) row of the chunk (without a dedicated for loop)
        # https://stackoverflow.com/questions/13682044/pandas-dataframe-remove-unwanted-parts-from-strings-in-a-column 
        df['lat'] = df['lat'].map(lambda x: str(x)[:2]+'.'+str(x)[3:])
        df['lon'] = df['lon'].map(lambda x: str(x)[:1]+'.'+str(x)[2:])
        
        # Cast the lon and lat columns from string to float64
        # the dates to datetime 
        # https://stackoverflow.com/questions/15891038/pandas-change-data-type-of-columns
        df['lat']   = pd.to_numeric(df['lat'],   errors='coerce', downcast='float')
        df['lon']   = pd.to_numeric(df['lon'],   errors='coerce', downcast='float')
        df['date']  = pd.to_datetime(df['date'], errors='coerce')
        df['linea'] = pd.to_numeric(df['linea'], errors='ignore', downcast='unsigned')# meglio lasciarla text, visto che ci sono linee con la lettera (es notturni etc)
        df.index += index_start

        linee_histo = find_histogram_class_from_unicode(df['linea'])
        val_linee = linee_histo[0]
        occ_linee = linee_histo[1]
        print 'Tot linee:', len(val_linee), ' per il chunk', cont
        #print  val_linee, occ_linee

        for i in xrange(0,len(val_linee)):

            linea = val_linee[i]#u'36  '
            print linea
            csv_database_linea = create_engine('sqlite:///..///databases///csv_database_formatted_boh_prova'+linea+'2.db')
            mask = df['linea'] == linea
            iso_linea = df[mask]
            #The day of the week with Monday=0, Sunday=6
            iso_linea['weekday'] = iso_linea['date'].dt.dayofweek
            iso_linea.to_sql('fcd_table', csv_database_linea, if_exists='append', index = False)
            #print mask
        cont+=1
        #print '{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, cont*chunksize)
        # sembrerebbe che si debba fare una run prima con if_exists='replace' e poi con if_exists='append'
        # ogni volta che si cambia il nome del database o una sua caratteristica
        # https://stackoverflow.com/questions/43607206/issue-with-saving-undefined-table-columns-with-pandas-to-sql
        #df.to_sql('fcd_table', csv_database, if_exists='append', index = False)
        index_start = df.index[-1] + 1
        if cont == 1: #fermo la creazione del database al quinto chunk
            sys.exit()
        #print df.columns.value

print 'Database created in', (dt.datetime.now() - start).seconds, 'seconds'