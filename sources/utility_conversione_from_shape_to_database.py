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

plt.close('all')

# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import cv2

path = os.path.dirname(os.path.abspath(__file__))+'//..//'


route_data = np.genfromtxt(path+"\\databases\\velocities_nuove\\shape\\shape_velocities_line_39.csv", delimiter = ';', dtype = object)
route_data[1:,7] = route_data[1:,7].astype(np.float)
route_data[1:,8] = route_data[1:,8].astype(np.float) 
route_data[1:,-1] = route_data[1:,-1].astype(np.int)  
print route_data
print len(route_data)
route_data = route_data[ route_data[:,-1] == 0]
print len(route_data)


plt.figure()
plt.plot(route_data[1:,7], route_data[1:,8],'bo')
plt.show()

sys.exit()


sf = shp.Reader(path+"\\databases\\velocities_nuove\\shape\\shape_velocities_line_39.shp")
print sf


for j in xrange (len(sf.shapeRecords())):
    #X = []
    #Y = []
    XY = []
    for shape in sf.shapeRecords():
        #IDlinea = shape.record[2][25:-9]
        #print 'linea', IDlinea,  shape.record[1][6:]
        for index in range(len(shape.shape.points)):
            print float(index) *100.0/len(shape.shape.points)
            if shape.record[-1] == 0: # non è un outlieer
                XY.append([shape.shape.points[index][0], shape.shape.points[index][1]])
            #pdb.set_trace()

# If not already present, create the directory to store the downloaded data
database_directory = 'routes_databases'
if not os.path.exists(path+'//databases//'+database_directory):
    os.makedirs(path+'//databases//'+database_directory)
            
'''df = pd.DataFrame(XY)
df.columns = ['lon', 'lat']
database_route_linea = create_engine('sqlite:///..///databases///'+database_directory+'///routes_database_no_outlier'+IDlinea+'.db')
df.to_sql('route_table', database_route_linea, if_exists='append', index = False)
            
sys.exit()'''