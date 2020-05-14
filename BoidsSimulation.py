import random
import pyglet
import numpy as np 
from pyglet.gl import (
    Config,
    glEnable, glBlendFunc, glLoadIdentity, glClearColor,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_COLOR_BUFFER_BIT)
from pyglet.window import key
from World import *

#Config
numBoids = 150
walls = 1 # 0 = no walls, 1 = walls
rangeClustering = 2 # 0=none, 1=DBSCAN, 2=Tiling, 3=DBSCAN w/ Tiling
reclusterNum = 20
tileWidth = 40
clusterIndicators = 0 #0 = no indicator, 1 = indicators

boidRange = tileWidth ** 2
boidCollisionRange = 14 ** 2
boidCollisionWeight = 4
boidVelMatchingWeight = 0.5
boidFlockCenteringWeight = 0.3
boidWallRange = 60
boidwalAvoidWeight = 5000
boidMinSpeed = 45
boidMaxSpeed = 60
boidSize = 6
boidViewAngle = 290 * (math.pi/180)


boidConfig = [boidRange, boidCollisionRange, boidCollisionWeight, boidVelMatchingWeight, boidFlockCenteringWeight, boidwalAvoidWeight, boidMinSpeed, boidMaxSpeed, boidSize, boidViewAngle, walls, boidWallRange]
#640, 360
world = World(640, 360, numBoids, boidConfig, rangeClustering, reclusterNum, tileWidth, clusterIndicators);

window = pyglet.window.Window(640, 360,
    fullscreen=False,
    caption="Boids Simulation")

glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

fps_display = pyglet.window.FPSDisplay(window=window)

def update(dt):
    world.updateLocalBoids()
    world.updateBoidPos(1/15)

# schedule world updates as often as possible
pyglet.clock.schedule(update)


@window.event
def on_draw():
    glClearColor(0.1, 0.1, 0.1, 1.0)
    window.clear()
    glLoadIdentity()

    #fps_display.draw()

    batch = pyglet.graphics.Batch()
    vl = world.getVetexBatch()
    cl = world.getColourBatch()
    for i in range(0, len(vl)):
        batch.add(3, pyglet.gl.GL_TRIANGLES, None, ('v2f', (vl[i][0], vl[i][1], vl[i][2], vl[i][3], vl[i][4], vl[i][5])), ('c3B', (cl[i][0], cl[i][1], cl[i][2], cl[i][0], cl[i][1], cl[i][2], cl[i][0], cl[i][1], cl[i][2])))


@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.Q:
        sns.lineplot(x=np.arange(0,len(fps)), y=fps)
        matplotlib.pyplot.show() 
        pyglet.app.exit()
        self.close()
    

pyglet.app.run()

