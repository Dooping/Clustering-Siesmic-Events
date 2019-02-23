# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:11:23 2016

@author: Bruno Ferreira, David Gago, Emidio Correio
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from scipy.misc import imread
from sklearn.cluster import KMeans,DBSCAN

# plot 3d dos clusters
def plot_classes_3d(labels,data):       
    plt.figure(figsize=(10,10))    
    plt.subplot(111, projection="3d")       
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(data[mask,0], data[mask,1], data[mask,2],'o', markersize=4, mew=1,zorder=1,alpha=0.3)
        ix = ix+1                    
    mask = np.logical_not(nots)    
    if np.sum(mask)>0:
        plt.plot(data[mask,0], data[mask,1],data[mask,2], 'k.', markersize=1, mew=1,markerfacecolor='w',zorder=1)
    plt.axis('on')

def plot_classes_mw(labels,lon,lat):
    """Plot seismic events using Mollweide projection.
    Arguments are the cluster labels and the longitude and latitude
    vectors of the events"""
    img = imread("Mollweide_projection_SW.jpg")        
    plt.figure(figsize=(10,5))    
    plt.subplot(111, projection="mollweide")
    plt.imshow(img,zorder=0,extent=[-np.pi,np.pi,-np.pi/2,np.pi/2],aspect=0.5)        
    nots = np.zeros(len(labels)).astype(bool)
    diffs = np.unique(labels)    
    ix = 0
    x = lon/180*np.pi
    y = lat/180*np.pi
    for lab in diffs[diffs>=0]:        
        mask = labels==lab
        nots = np.logical_or(nots,mask)        
        plt.plot(x[mask], y[mask],'o', markersize=4, mew=1,zorder=1,alpha=0.5)
        ix = ix+1                    
    mask = np.logical_not(nots)    
    if np.sum(mask)>0:
        plt.plot(x[mask], y[mask], 'k.', markersize=1, mew=1,markerfacecolor='w',zorder=1)
    plt.axis('off')

#obter o k-esimo vizinho, vetor ordenado 
def knearneighbours(k):
    neigh = KNeighborsClassifier(k)
    neigh.fit(data3d,np.zeros(data3d.shape[0]))
    dist, ind = neigh.kneighbors()
    dist = dist[:,-1]#Kth neighbour
    
    ordered = np.sort(dist) #ordenar
    return ordered

#metodo inicial de obtencao do eps - legacy
def thresholdMethod(th, orderedDist):
    normalized = orderedDist/orderedDist[-1]
    gradient = np.diff(normalized)*normalized.shape[0]
    indices = np.where(gradient > th)[0]
    return orderedDist[indices[0]],indices[0]

#metodo de obtencao do eps como a professora sugeriu
def orderedGradientMethod(th, orderedDist,n):
    normalized = orderedDist/orderedDist[-1]#normalizar
    gradient = np.diff(normalized,n)*normalized.shape[0]*n#derivada
    sort_index = np.argsort(gradient)#guardar os indices do vetor ordenado para conseguir recuperar
    gradient = np.sort(gradient)#ordenar vetor de derivadas
    idx = np.searchsorted(gradient,th)#encontrar primeiro acima de th(i.e. th=1)
    distIdx = np.searchsorted(sort_index,idx)#recuperar o indice no vetor de distancias
    return orderedDist[distIdx],distIdx
  
#teste ao DBSCAN com os parametros dados  
def DBSCANtest(eps, k):
    db = DBSCAN(eps, min_samples=k).fit(data3d)
    return silhouette_score(data3d,db.labels_)  
    
    
    
###########################################################
# load data
data = pd.read_csv('century6_5.csv')
lat = data.latitude.values
lon = data.longitude.values

#show data
'''
plt.figure(figsize = (10,5))
plt.plot(lon,lat,'.')
plt.show()
'''

# transformar lat,lon em x,y,z
RADIUS = 6371
x = RADIUS * np.cos(lat * np.pi/180) * np.cos(lon * np.pi/180)
y = RADIUS * np.cos(lat * np.pi/180) * np.sin(lon * np.pi/180)
z = RADIUS * np.sin(lat * np.pi/180)

#show world
'''
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x, y, z, s=10)
plt.show()
'''

#juntar num array de features
data3d = np.zeros((x.shape[0],3))
data3d[:,0] = x
data3d[:,1] = y
data3d[:,2] = z

#teste do k-means
kmeans_min_k = 14
kmeans_max_k = 15
silh_score = []
bestK = 0
bestSilh = -1
bestLabels = []
xArray = []

for k in range(kmeans_min_k,kmeans_max_k):
    kmeans = KMeans(k, random_state = 0).fit(data3d)
    labels = kmeans.predict(data3d)
    silh = silhouette_score(data3d,labels)
    if (silh > bestSilh):
        bestSilh = silh
        bestK = k
        bestLabels = labels
    silh_score.append(silh)
    xArray.append(k)
    
#plot do grafico 1
plt.figure(figsize=(10,10)) 
plt.plot(xArray, silh_score, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.grid(True)
plt.show()

#imprimir resultados
print 'Best k (number of clusters): ', bestK, ' silhouette score: ', bestSilh

#plot clusters com o melhor k-means, figura 1
plot_classes_mw(bestLabels, lon, lat)


#teste do DBSCAN
dbscan_min_k = 1
dbscan_max_k = 30
bestKnn = 0
bestScore = -1
bestEps = 0
bestEpsIdx = 0
bestKList = []
scoreArray = []
xxArray = []

threshold = 1#declive onde se obtem eps
n = 1#numero de pontos a considerar na derivada(alem do proprio)

for k in range(dbscan_min_k,dbscan_max_k):
    dist = knearneighbours(k)
    eps,idx = orderedGradientMethod(threshold,dist,n)
    score = DBSCANtest(eps, k)
    scoreArray.append(score)
    xxArray.append(k)
    if (score > bestScore):#guardar os parametros do melhor
        bestScore = score
        bestKnn = k
        bestEps = eps
        bestEpsIdx = idx
        bestKList = dist

#plot do calculo do eps, grafico 2
plt.figure(figsize=(10,10))
plt.plot(range(0, bestKList.shape[0]), bestKList)
plt.plot(bestEpsIdx,bestEps, marker='o', color='r')
plt.grid(True)
plt.xlabel('Point index')
plt.ylabel('Distance (km)')
plt.show()  

#plot do grafico 3
plt.figure(figsize=(10,10))
plt.plot(xxArray, scoreArray, marker='o')
plt.grid(True)
plt.xlabel('Number of neighbours')
plt.ylabel('Silhouette score')
plt.show()       
        
db = DBSCAN(bestEps, min_samples=bestKnn).fit(data3d)

#plot do melhor DBSCAN, figura 2
plot_classes_mw(db.labels_, lon, lat)

#plot do melhor DBSCAN em 3d
plot_classes_3d(db.labels_, data3d)

#plot do melhor resultado
print 'Best number of neighbours: ', bestKnn, ' eps value: ', bestEps, ' silhouette score: ', bestScore

