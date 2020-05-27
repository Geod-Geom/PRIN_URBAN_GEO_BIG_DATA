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
plt.close('all')
#rc('font', weight='bold')

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


def _button_clicked(event):

    global pause, bnext
    if pause == False:
        anim.event_source.stop()
        pause = True
        bnext.label.set_text("RUN") # works
    else:
        anim.event_source.start()
        pause = False
        bnext.label.set_text("STOP")

pause = False
path = os.path.dirname(os.path.abspath(__file__))+'//..//'
line = 39
mezzo = 309#2581
data_stops = read_db(path+'databases//routes_databases//routes_database_'+str(line)+'.db','route_table')
# conversion to web mercator coordinates (E,N from phi and lambda)
# https://gis.stackexchange.com/questions/247871/convert-gps-coordinates-to-web-mercator-epsg3857-using-python-pyproj
for index in range (len(data_stops)):
    data_stops[index, 0], data_stops[index, 1] = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), data_stops[index, 0], data_stops[index, 1])


results = np.genfromtxt (path+'//Risultati_proiezione//linea_'+str(line)+'_mezzo_'+str(mezzo)+'.txt', skip_header =1, dtype = object)
#results = results[40:, :]
results[:,0:3] = results[:,0:3].astype(np.float)
results[:, -4:] = results[:, -4:].astype(np.float)

print results

minE = min(np.min (results[:,1]), np.min (results[:,-2]), np.min(data_stops[:,0]))
maxE = max(np.max (results[:,1]), np.max (results[:,-2]), np.max(data_stops[:,0]))
minN = min(np.min (results[:,2]), np.min (results[:,-1]), np.min(data_stops[:,1]))
maxN = max(np.max (results[:,2]), np.max (results[:,-1]), np.max(data_stops[:,1]))

print 'min E', minE
print 'max E', maxE
print 'min N', minN
print 'max N', maxN


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()

axnext = plt.axes([0.9, 0.0, 0.1, 0.075])
bnext = Button(axnext, 'STOP', color='white')
bnext.on_clicked(_button_clicked)



#bnext.on_clicked(callback.next)

ax = plt.axes(xlim=(minE-1000, maxE+1000), ylim=(minN-1000, maxN+1000), )
plt.plot(data_stops[:,0],data_stops[:,1], 'ko--')
for zz in range(len(data_stops)):
        plt.text(data_stops[zz,0]-0.00001, data_stops[zz,1]-0.00001, str(zz) , color='k',fontsize=12 )

plt.xlabel('E [m]', fontweight='bold')
plt.ylabel('N [m]', fontweight='bold')
ax.ticklabel_format(useOffset=False, style='plain')
plt.axis('equal')


ax.grid()
projected_points, = ax.plot([], [], 'ro', lw=2, )
raw_points, = ax.plot([], [], 'bo', lw=2, )
#projected_text = ax.text([], [],'',  color='r', fontsize=12, weight='bold' )


number_of_points_displayed = 10
# contengono le label dei punti
raw_texts = []
projected_texts = []
for index in range (number_of_points_displayed):
    raw_texts.append(ax.text([], [],'',  color='b', fontsize=12, weight='bold' ))
    projected_texts.append(ax.text([], [],'',  color='r', fontsize=12, weight='bold' ))


# initialization function: plot the background of each frame
def init():
    projected_points.set_data([], [])
    raw_points.set_data([], [])
   
    return projected_points, raw_points


# animation function. This is called sequentially
def animate(i):

    print 'new frame', i

    #pdb.set_trace()
    if i > (number_of_points_displayed-1):
        projected_points.set_data( results[i-number_of_points_displayed+1:i+1,1], results[i-number_of_points_displayed+1:i+1,2])# grafico il punto all'indice i e tutti i precedenti
        raw_points.set_data( results[i-number_of_points_displayed+1:i+1,-2], results[i-number_of_points_displayed+1:i+1,-1])# grafico il punto all'indice i e tutti i precedenti
    else:
        projected_points.set_data( results[:i+1,1], results[:i+1,2])# grafico il punto all'indice i e tutti i precedenti
        raw_points.set_data( results[:i+1,-2], results[:i+1,-1])# grafico il punto all'indice i e tutti i precedenti
    
    
    for boh in range (len(raw_texts)):
        for zz in range (i+1):
                if zz- boh >= 0:
                    raw_texts[boh].set_x(results[zz -boh,-2]-0.00001)
                    raw_texts[boh].set_y(results[zz- boh,-1]-0.00001)
                    raw_texts[boh].set_text(str(zz -boh))#mon ci possono essere 2 raw text contemporaneamente: devo usare vettori di raw text
                    projected_texts[boh].set_x(results[zz -boh,1]+0.00006)
                    projected_texts[boh].set_y(results[zz -boh,2]+0.00006)
                    projected_texts[boh].set_text(str(zz -boh)+'   '+results[zz -boh,4] )
    return projected_points, raw_points, projected_texts, raw_texts


# # call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, range(len(results)), init_func=init, interval=1500, blit=False)

plt.show()