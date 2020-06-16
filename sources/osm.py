#!/usr/bin/python
# -*- coding: utf-8 -*-
# https://github.com/gboeing/osmnx/blob/master/examples/12-node-elevations-edge-grades.ipynb
#http://geoffboeing.com/2016/11/osmnx-python-street-networks/
import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg')
print '#1 Backend:',plt.get_backend()
import numpy as np

import osmnx as ox
import networkx as nx

city = ox.gdf_from_place('Turin,IT')
#G = ox.graph_from_place('Torino, Italy')

#G = ox.graph_from_point((7.621083300000009, 45.0743356), distance=0.750, network_type='drive')

G = ox.graph_from_address('Corso Francia, Torino, Italia', network_type='drive', name = 'Torino')

#ox.plot_shape(ox.project_gdf(city))
fig, ax =ox.plot_graph(G, close = False, show=False, axis_off=False)# è importante metter false
ax.plot(7.64947,45.0749,'ro',markersize=5)
fig.canvas.draw()
#plt.plot(7.64947,45.0749,'o',markersize=50)

plt.xlabel('longitude')
plt.ylabel('latitude')
plt.savefig('torino.png')
plt.show()



'''x_inf = -122.43
x_sup = -122.41
y_inf = 37.78
y_sup = 37.79'''

x_inf = 7.51
x_sup = 7.68
y_inf = 45.035
y_sup = 45.080

G = ox.graph_from_bbox(y_sup, y_inf, x_sup, x_inf, network_type='drive', name = 'prova')
# it also possible to project the network to UTM (zone calculated automatically) then plot it
#G_projected = ox.project_graph(G)

origin_point = (7.621083300000009, 45.0743356)
origin_node = ox.get_nearest_node(G, origin_point)

destination_point = (7.521929999999998, 45.07037)
destination_node = ox.get_nearest_node(G, destination_point) 
# find the route between these nodes then plot it
route = nx.shortest_path(G, origin_node, destination_node)
fig, ax = ox.plot_graph_route(G, route, origin_point=origin_point, destination_point=destination_point)

fig, ax =ox.plot_graph(G, close = False, show=False, axis_off=False)
ax.plot(7.64947,45.0749,'ro',markersize=5)
plt.xlabel('longitude')
plt.ylabel('latitude')

plt.show()
#7.64947,45.0749

'''a = np.array(G.nodes())

place = 'portland'
point = (45.517309, -122.682138)
fig, ax = ox.plot_figure_ground(point=point, filename=place, show=False, save=True, close=False)
ax.plot(45.517309, -122.682138, markersize=18, markeredgewidth=2,markeredgecolor='k')
plt.show()'''