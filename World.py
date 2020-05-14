from Boid import *
from random import randint
import random
import numpy as np
from itertools import repeat
from sklearn.cluster import DBSCAN

class World(object):

    def __init__(self, width, height, numBoids, boidConfig, clustering, reclusterNum, tileWidth, clusterIndicators):
        random.seed(1)
        self.boids = self.CreateBoids(width, height, numBoids, boidConfig)
        self._numBoids = numBoids
        self._colourList = [[255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255], [255,127,0], [127,255,0], [127,0,255], [255,0,127], [0,127,255], [0,255,127], [127,0,0], [0,127,0], [0,0,127], [127,127,0], [127,0,127], [0,127,127], [127,127,127], [63,127,0], [63,0,127], [0,63,127], [127,63,0], [127,0,63], [0,127,63], [190,127,0], [190,0,127], [0,190,127], [127,190,0], [127,0,190], [0,127,190], [190,63,0], [190,0,63], [0,190,63], [63,190,0], [63,0,190], [0,63,190], [63,255,0], [63,0,255], [0,63,255], [255,63,0], [255,0,63], [0,255,63], [190,255,0], [190,0,255], [0,190,255], [255,190,0], [255,0,190], [0,255,190], [64,64,64], [190,190,190], [127,127,127], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255], [255,127,0], [127,255,0], [127,0,255], [255,0,127], [0,127,255], [0,255,127], [127,0,0], [0,127,0], [0,0,127], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255], [255,127,0], [127,255,0], [127,0,255], [255,0,127], [0,127,255], [0,255,127], [127,0,0], [0,127,0], [0,0,127], [127,127,0], [127,0,127], [0,127,127], [127,127,127], [63,127,0], [63,0,127], [0,63,127], [127,63,0], [127,0,63], [0,127,63], [190,127,0], [190,0,127], [0,190,127], [127,190,0], [127,0,190], [0,127,190], [190,63,0], [190,0,63], [0,190,63], [63,190,0], [63,0,190], [0,63,190], [63,255,0], [63,0,255], [0,63,255], [255,63,0], [255,0,63], [0,255,63], [190,255,0], [190,0,255], [0,190,255], [255,190,0], [255,0,190], [0,255,190], [64,64,64], [190,190,190], [127,127,127], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255], [255,127,0], [127,255,0], [127,0,255], [255,0,127], [0,127,255], [0,255,127], [127,0,0], [0,127,0], [0,0,127], [255,255,255]]
        self._clusterIndicators = clusterIndicators
        self.count = 0
        self.updateGroups = 1
        self.rangeClustering = clustering
        self.boidConfig = boidConfig
        self.reclusterNum = reclusterNum
        self.tileWidth = tileWidth
        self.width = width
        self.height = height

    def updateLocalBoids(self):

        if self.rangeClustering == 0:

            for i in range(0, len(self.boids)-1):
                for k in range(i+1, len(self.boids)):
                    difference = [self.boids[i]._position[0] - self.boids[k]._position[0], self.boids[i]._position[1] - self.boids[k]._position[1]]
                    if difference[0]**2 + difference[1]**2 < self.boidConfig[0]:
                        if self.boids[i].angleBetweenBoids(self.boids[i]._velocity, difference) < self.boids[i]._viewAngle:
                            self.boids[i]._boidMask[k] = 1 
                        if self.boids[k].angleBetweenBoids(self.boids[k]._velocity, difference) < self.boids[k]._viewAngle:
                            self.boids[k]._boidMask[i] = 1 

        elif self.rangeClustering == 1:

            # Compute DBSCAN
            X = self.getLocationBatch(self.boids)
            db = DBSCAN(eps=self.tileWidth, min_samples=2).fit(X)
            labels = db.labels_
            n_clusters_ = len(set(labels))

            groups = [[] for i in repeat(None, n_clusters_)]
            for i in range(0, len(self.boids)):
                if self._clusterIndicators == 1:
                    self.boids[i].setColour(self._colourList[labels[i]])
                if labels[i] != -1:
                    groups[labels[i]].append(i)

            for i in range(n_clusters_):            
                for k in range(0, len(groups[i])-1):
                    for j in range(k+1, len(groups[i])):
                        difference = [self.boids[groups[i][k]]._position[0] - self.boids[groups[i][j]]._position[0], self.boids[groups[i][k]]._position[1] - self.boids[groups[i][j]]._position[1]]
                        if difference[0]**2 + difference[1]**2 < self.boidConfig[0]:
                            if self.boids[groups[i][k]].angleBetweenBoids(self.boids[groups[i][k]]._velocity, difference) < self.boids[groups[i][k]]._viewAngle:
                                self.boids[groups[i][k]]._boidMask[groups[i][j]] = 1 
                            if self.boids[groups[i][j]].angleBetweenBoids(self.boids[groups[i][j]]._velocity, difference) < self.boids[groups[i][j]]._viewAngle:
                                self.boids[groups[i][j]]._boidMask[groups[i][k]] = 1 

        elif self.rangeClustering == 2:

            numTilesW = int(self.width/self.tileWidth)
            numTilesH = int(self.height/self.tileWidth)
            numTiles = numTilesW * numTilesH
            tiles = [[] for i in range(0, numTiles)]
            for i in range(0, len(self.boids)):
                x = int(self.boids[i]._position[0] // (self.width / numTilesW))
                x = np.min([numTilesW-1,x])
                y = int(self.boids[i]._position[1] // (self.height / numTilesH))
                y = np.min([numTilesH-1,y])
                tiles[x*numTilesH+y].append(i)
                if self._clusterIndicators == 1:
                    self.boids[i].setColour(self._colourList[(x*numTilesH+y)%len(self._colourList)])

            self.boids[0].setColour([255,0,0])
            x = int(self.boids[0]._position[0] // (self.width / numTilesW)) * (self.width / numTilesW)
            y = int(self.boids[0]._position[1] // (self.height / numTilesH)) * (self.height / numTilesH)
                  
            for x in range(0,numTilesW):
                for y in range(0,numTilesH):
                    for i in tiles[x*numTilesH+y]:
                        self.boids[i]._boidMask[tiles[x*numTilesH+y]] = 1
                        if x > 0:
                            self.boids[i]._boidMask[tiles[(x-1)*numTilesH+y]] = 1
                        if x < numTilesW-1:
                            self.boids[i]._boidMask[tiles[(x+1)*numTilesH+y]] = 1
                        if y > 0:
                            self.boids[i]._boidMask[tiles[x*numTilesH+(y-1)]] = 1
                        if y < numTilesH-1:
                            self.boids[i]._boidMask[tiles[x*numTilesH+(y+1)]] = 1

                        if x > 0 and y > 0:
                            self.boids[i]._boidMask[tiles[(x-1)*numTilesH+y-1]] = 1
                        if x > 0 and y < numTilesH-1:
                            self.boids[i]._boidMask[tiles[(x-1)*numTilesH+y+1]] = 1
                        if x < numTilesW-1 and y > 0:
                            self.boids[i]._boidMask[tiles[(x+1)*numTilesH+y-1]] = 1
                        if x < numTilesW-1 and y < numTilesH-1:
                            self.boids[i]._boidMask[tiles[(x+1)*numTilesH+y+1]] = 1

            #unflag boids out of range
            for i in range(0,len(self.boids)):
                self.boids[i]._boidMask[i] = 0
                indexs = np.where(self.boids[i]._boidMask == 1)
                if np.size(indexs) > 0:
                    for indx in np.nditer(indexs):
                        difference = [self.boids[i]._position[0] - self.boids[indx]._position[0], self.boids[i]._position[1] - self.boids[indx]._position[1]]
                        if (difference[0]**2 + difference[1]**2 > self.boidConfig[0]) or (self.boids[i].angleBetweenBoids(self.boids[i]._velocity, difference) > self.boids[i]._viewAngle):
                            self.boids[i]._boidMask[indx] = 0

        elif self.rangeClustering == 3:
             # Compute DBSCAN
            X = self.getLocationBatch(self.boids)
            db = DBSCAN(eps=self.tileWidth, min_samples=2).fit(X)
            labels = db.labels_
            n_clusters_ = max(labels) + 1

            groups = [[] for i in repeat(None, n_clusters_)]
            clusterBounds = np.zeros((n_clusters_, 4))
            for i in range(0, n_clusters_): clusterBounds[i,:] = [self.width,0,self.height,0] #minX, maxX, minY, maxY
            for i in range(0, len(self.boids)):
                if self._clusterIndicators == 1:
                    self.boids[i].setColour(self._colourList[labels[i]])
                if labels[i] != -1:
                    groups[labels[i]].append(i)
                    if self.boids[i]._position[0] < clusterBounds[labels[i],0]:
                        clusterBounds[labels[i],0] = self.boids[i]._position[0]
                    if self.boids[i]._position[0] > clusterBounds[labels[i],1]:
                        clusterBounds[labels[i],1] = self.boids[i]._position[0] + 1
                    if self.boids[i]._position[1] < clusterBounds[labels[i],2]:
                        clusterBounds[labels[i],2] = self.boids[i]._position[1]
                    if self.boids[i]._position[1] > clusterBounds[labels[i],3]:
                        clusterBounds[labels[i],3] = self.boids[i]._position[1] + 1
            
            for i in range(n_clusters_): #Tile for each cluster
                if len(groups[i]) < self.reclusterNum:
                    for k in range(0, len(groups[i])-1):
                        for j in range(k, len(groups[i])):
                            difference = [self.boids[groups[i][k]]._position[0] - self.boids[groups[i][j]]._position[0], self.boids[groups[i][k]]._position[1] - self.boids[groups[i][j]]._position[1]]
                            if difference[0]**2 + difference[1]**2 < self.boidConfig[0]:
                                if self.boids[groups[i][k]].angleBetweenBoids(self.boids[groups[i][k]]._velocity, difference) < self.boids[groups[i][k]]._viewAngle:
                                    self.boids[groups[i][k]]._boidMask[groups[i][j]] = 1 
                                if self.boids[groups[i][j]].angleBetweenBoids(self.boids[groups[i][j]]._velocity, difference) < self.boids[groups[i][j]]._viewAngle:
                                    self.boids[groups[i][j]]._boidMask[groups[i][k]] = 1 
                else:
                    width = (clusterBounds[i,1]-clusterBounds[i,0]) 
                    height = (clusterBounds[i,3]-clusterBounds[i,2])
                    numTilesW = int(width/self.tileWidth) + 1
                    numTilesH = int((height)/self.tileWidth) + 1
                    numTiles = numTilesW * numTilesH
                    tiles = [[] for i in range(0, numTiles)]
                    for k in range(0, len(groups[i])):
                        x = int((self.boids[groups[i][k]]._position[0] - clusterBounds[i,0]) // (width / numTilesW))
                        x = np.max([np.min([numTilesW-1,x]),0])
                        y = int((self.boids[groups[i][k]]._position[1]  - clusterBounds[i,2]) // (height / numTilesH))
                        y = np.max([np.min([numTilesH-1,y]),0])
                        tiles[x*numTilesH+y].append(groups[i][k])
                  
                    for x in range(0,numTilesW):
                        for y in range(0,numTilesH):
                            for i in tiles[x*numTilesH+y]:
                                self.boids[i]._boidMask[tiles[x*numTilesH+y]] = 1
                                if x > 0:
                                    self.boids[i]._boidMask[tiles[(x-1)*numTilesH+y]] = 1
                                if x < numTilesW-1:
                                    self.boids[i]._boidMask[tiles[(x+1)*numTilesH+y]] = 1
                                if y > 0:
                                    self.boids[i]._boidMask[tiles[x*numTilesH+(y-1)]] = 1
                                if y < numTilesH-1:
                                    self.boids[i]._boidMask[tiles[x*numTilesH+(y+1)]] = 1

                                if x > 0 and y > 0:
                                    self.boids[i]._boidMask[tiles[(x-1)*numTilesH+y-1]] = 1
                                if x > 0 and y < numTilesH-1:
                                    self.boids[i]._boidMask[tiles[(x-1)*numTilesH+y+1]] = 1
                                if x < numTilesW-1 and y > 0:
                                    self.boids[i]._boidMask[tiles[(x+1)*numTilesH+y-1]] = 1
                                if x < numTilesW-1 and y < numTilesH-1:
                                    self.boids[i]._boidMask[tiles[(x+1)*numTilesH+y+1]] = 1
                 
                    #unflag boids out of range               
                    self.boids[i]._boidMask[i] = 0
                    indexs = np.where(self.boids[i]._boidMask == 1)
                    if np.size(indexs) > 0:
                        for indx in np.nditer(indexs):
                            difference = [self.boids[i]._position[0] - self.boids[indx]._position[0], self.boids[i]._position[1] - self.boids[indx]._position[1]]
                            if (difference[0]**2 + difference[1]**2 > self.boidConfig[0]) or (self.boids[i].angleBetweenBoids(self.boids[i]._velocity, difference) > self.boids[i]._viewAngle):
                                self.boids[i]._boidMask[indx] = 0
            
    def updateBoidPos(self, dt):
        for i in range(0, len(self.boids)):
            self.boids[i].updatePos(dt, self.boids)

    def CreateBoids(self, width, height, num, boidConfig):
        boids = []
        for i in range(num):
            x = randint(boidConfig[6],boidConfig[7])
            if randint(0,1) == 1:
                x = -x
            y = randint(boidConfig[6],boidConfig[7])
            if randint(0,1) == 1:
                y = -y
            #boids.append(Boid([randint(0,width), randint(0,height)], [width,height], [x, y], [255, 255, 255], num, boidConfig))
            boids.append(Boid([width/2, height/2], [width,height], [x, y], [255, 255, 255], num, boidConfig))
        return boids

    def getVetexBatch(self):
        batch = []
        for b in self.boids:
            batch.append(b.getVertexList())
        return batch

    def getColourBatch(self):
        batch = []
        for b in self.boids:
            batch.append(b._colour)
        return batch

    def getLocationBatch(self, boids):
        batch = []
        for b in boids:
            batch.append(b._position)
        return batch
        
