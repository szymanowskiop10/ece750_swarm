import copy
import random
import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from maze import Maze, SENSORRANGE
from swarm import MobileRobot, Swarm

frame_dir = "frames"
os.makedirs(frame_dir, exist_ok=True)

num_robot1 = 1500
height1 = 8
width1 = 7
source1 = [5.25, 3.75]

num_robot2 = 3000
height2 = 15
width2 = 10
source2 = [0.25, 13.75]

test_small: bool = False

TIME_INTERVAL = 0.01
arrow_offset = np.array([[.025, .0], [.0, .025], [-.025, .0], [.0, -.025]])
arrow_len = np.array([[-.05, .0], [.0, -.05], [.05, .0], [.0, .05]])

height = height1 if test_small else height2
width = width1 if test_small else width2
source = source1 if test_small else source2
num_robot = num_robot1 if test_small else num_robot2

print('simulation started, initializing swarm...')
swarm = Swarm(step_length=.01, t=.0)
print('adding {0} robots to the swarm'.format(num_robot))
swarm.add_robot_batch(num_robot, source)

print('building the maze...')
maze = Maze(height, width, 0.5)

fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_xticks(np.arange(0, width, 0.5))
# ax.set_yticks(np.arange(0, height, 0.5))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
# ax.grid()
ax.set_aspect(1)

def draw_maze(m: Maze, s: Swarm, source: List[float] = [.1, .1]):   
    # define shapes
    cirs = m.get_cirs()
    for circle in cirs:
        ax.add_patch(Circle(circle[0], circle[1], edgecolor='xkcd:grey', facecolor='xkcd:grey'))
    
    tris = m.get_tris()
    for triangle in tris:
        p1 = np.array([triangle[0][0], triangle[0][1]])
        p2 = np.array([triangle[1][0], triangle[1][1]])
        p3 = np.array([triangle[2][0], triangle[2][1]])
        p = np.array([p1, p2, p3])
        ax.add_patch(Polygon(p, edgecolor='xkcd:grey', facecolor='xkcd:grey'))

    ax.add_patch(Rectangle((source[0]-0.08, source[1]-0.08), 0.16, 0.16, edgecolor='xkcd:deep red', facecolor='xkcd:deep red'))

    # define swarms  
    count = s.get_num()
    for i in range(count):
        id = i + 1
        circle = s.get_geometry(id)
        dir = s.get_robot_dir(id)
        first_activated = s.get_activated_once(id)
        x_c, y_c = circle[0][0], circle[0][1]
        if first_activated:
            if dir != -1:
                ax.add_patch(Circle((x_c, y_c), circle[1], edgecolor=None, fill='xkcd:light blue'))
            else:
                ax.add_patch(Circle((x_c, y_c), circle[1], edgecolor='black', fill=None))

    # define survivors
    survivors = m.get_people()
    for person in survivors:
        x, y = person[0], person[1]
        p1 = np.array([x-0.08, y-0.04])
        p2 = np.array([x+0.08, y-0.04])
        p3 = np.array([x, y+0.0986])
        p = np.array([p1, p2, p3])
        ax.add_patch(Polygon(p, edgecolor='xkcd:deep red', facecolor='xkcd:deep red'))

    # plot the paths to survivors
    path = s.get_path_to_surv(m)
    if path:
        x_path = [p[0] for p in path]
        y_path = [p[1] for p in path]
        plt.plot(x_path, y_path, color='xkcd:deep red')
        ax.add_patch(Circle((x_path[-1], y_path[-1]), SENSORRANGE, edgecolor='xkcd:deep red', fill=None))

if __name__ == '__main__':
    if test_small:
        # adding obstacles, walls and the single survivor
        print('adding walls')
        maze.add_rect(.0, .0, .1, 8.0)
        maze.add_rect(.0, .0, 7.0, .1)
        maze.add_rect(.0, 7.9, 7.0, 8.0)
        maze.add_rect(6.9, .0, 7.0, 8.0)
        maze.add_rect(4.6, 3.1, 6.6, 3.45)
        maze.add_rect(4.6, 3.1, 4.95, 5.4)
        maze.add_rect(4.6, 5.05, 6.6, 5.4)
        maze.add_rect(6.25, 3.1, 6.6, 4.0)
        maze.add_rect(6.25, 4.5, 6.6, 5.4)
        maze.add_rect(3.2, .0, 3.4, 1.1)
        maze.add_rect(3.2, 1.6, 3.4, 2.1)
        maze.add_rect(.7, 3.65, 4.6, 3.8)
        maze.add_rect(.7, 1.6, .85, 3.8)
        maze.add_rect(.7, 1.6, 2.6, 1.75)
        maze.add_rect(2.9, 1.6, 3.2, 1.75)
        maze.add_rect(1.6, 1.6, 1.75, 3.1)
        maze.add_rect(.0, .6, .6, .75)
        maze.add_rect(.0, 4.6, 1.0, 4.8)
        maze.add_rect(3.1, 5.25, 4.6, 5.4)
        maze.add_rect(3.1, 4.4, 3.25, 5.4)
        maze.add_rect(3.1, 3.75, 3.24, 4.1)
        maze.add_rect(2.2, 3.7, 2.35, 5.14)
        maze.add_rect(4.6, .0, 4.75, 1.8)
        maze.add_rect(5.4, 1.65, 7.0, 1.8)
        maze.add_rect(5.55, .4, 5.7, 1.8)
        maze.add_rect(5.4, .4, 6.3, .7)
        maze.add_rect(6.06, 1.8, 6.21, 2.64)
        maze.add_rect(5.08, 2.36, 5.23, 3.2)

        maze.add_tri(4.6, 5.4, 4.85, 5.4, 3.1, 8.0)
        maze.add_tri(4.85, 5.4, 3.1, 8.0, 3.3, 8.0)
        maze.add_tri(3.2, 2.1, 3.4, 2.1, 4.6, 3.1)
        maze.add_tri(3.2, 2.1, 4.6, 3.1, 4.6, 3.5)
        maze.add_tri(.7, 1.6, 2.6, 1.6, 1.8, .4)
        maze.add_tri(3.2, 1.1, 3.2, .0, 2.3, .0)
        maze.add_tri(2.7, 6.7, 2.9, 6.7, 1.0, 4.8)
        maze.add_tri(2.7, 6.7, .8, 4.8, 1.0, 4.8)
        maze.add_tri(2.7, 6.7, 2.9, 6.7, 2.8, 6.83)
        maze.add_tri(1.0, 4.6, 1.0, 4.8, 1.25, 5.2)
        maze.add_tri(.0, 7.9, 3.1, 7.9, 1.9, 7.3)
        maze.add_tri(6.196, 6.1, 6.446, 6.1, 5.1, 8.0)
        maze.add_tri(6.446, 6.1, 5.1, 8.0, 5.3, 8.0)
        maze.add_tri(4.71, 7.1, 5.1, 7.24, 5.5, 6.1)
        maze.add_tri(5.35, 6.3, 5.5, 6.3, 5.5, 6.1)
        maze.add_tri(.1, 7.9, 1.3, 7.9, 1.2, 7.31)

        maze.add_cir(.9, 6.1, .5)

        print('adding the survivor')
        maze.add_surv(1.2, 1.937)

    else:
        print('adding walls')
        maze.add_rect(.0, .0, .15, 12.5)
        maze.add_rect(.0, .0, 10.0, .1)
        maze.add_rect(9.85, .0, 10.0, 15.0)
        maze.add_rect(.0, 14.9, 10.0, 15.0)
        maze.add_rect(1.25, .0, 1.6, 2.5)
        maze.add_rect(1.25, 2.15, 2.3, 2.5)
        maze.add_rect(3.0, 2.15, 3.75, 2.5)
        maze.add_rect(3.4, .0, 3.75, 2.5)
        maze.add_rect(.0, 3.15, 3.75, 3.5)
        maze.add_rect(.0, 8.5, 3.75, 8.85)
        maze.add_rect(.0, 12.15, 3.75, 12.5)
        maze.add_rect(3.4, 8.5, 3.75, 11.0)
        maze.add_rect(3.4, 11.5, 3.75, 12.5)
        maze.add_rect(3.4, 3.15, 3.75, 6.0)
        maze.add_rect(3.4, 6.5, 3.75, 8.5)
        maze.add_rect(4.75, 11.53, 10, 11.88)
        maze.add_rect(4.75, 8.9, 8.0, 9.25)
        maze.add_rect(8.5, 8.9, 10.0, 9.25)
        maze.add_rect(4.75, 8.9, 5.1, 10.5)
        maze.add_rect(4.75, 11.0, 5.1, 11.88)
        maze.add_rect(6.75, 8.9, 7.1, 9.5)
        maze.add_rect(6.75, 10.0, 7.1, 11.88)
        maze.add_rect(7.4, 6.4, 7.75, 6.75)
        maze.add_rect(7.4, 6.4, 7.75, 8.9)
        maze.add_rect(7.4, 6.4, 10.0, 6.75)
        maze.add_rect(7.4, 7.55, 8.5, 7.9)
        maze.add_rect(9.0, 7.55, 10.0, 7.9)
        maze.add_rect(8.5, 4.5, 10.0, 4.85)
        maze.add_rect(.0, 5.0, 2.9, 5.2)
        maze.add_rect(4.9, .0, 5.25, 2.0)

        maze.add_cir(1.25, 4.0, 0.4)
        maze.add_cir(1.25, 5.9, 0.4)
        maze.add_cir(1.25, 7.8, 0.4)
        maze.add_cir(1.25, 9.7, 0.4)
        maze.add_cir(1.25, 11.6, 0.4)
        maze.add_cir(5.6, 2.0, 0.7)
        maze.add_cir(7.55, 2.0, 0.7)

        maze.add_tri(7.1, 9.25, 8.0, 9.25, 8.0, 10.0)
        maze.add_tri(2.7, 13.1, 6.3, 14.0, 5.0, 14.4)
        maze.add_tri(2.7, 13.1, 1.9, 14.2, 4.25, 14.0)
        maze.add_tri(6.6, 3.7, 9.55, 4.6, 5.9, 6.4)
        maze.add_tri(7.1, 15.0, 7.3, 15.0, 8.4, 12.6)
        maze.add_tri(8.4, 12.6, 8.6, 12.8, 7.3, 15.0)
        maze.add_tri(5.5, 11.88, 5.7, 11.88, 6.6, 13.66)
        maze.add_tri(6.6, 13.66, 6.8, 13.77, 5.7, 11.88)
        maze.add_tri(7.2, 2.4, 10.0, .0, 10.0, 3.54)
        maze.add_tri(7.6, 1.5, 10.0, 2.5, 10.0, .0)
        maze.add_tri(7.1, 9.5, 7.1, 9.0, 8.0, 10.0)
        maze.add_tri(4.75, 4.37, 5.25, 4.37, 6.6, 6.6)
        maze.add_tri(6.4, 4.9, 6.7, 6.4, 6.6, 6.6)
        maze.add_tri(6.6, 6.6, 6.6, 6.0, 6.2, 6.0)
        maze.add_tri(6.25, 6.25, 5.4, 8.0, 5.6, 8.0)
        maze.add_tri(6.4, 6.25, 6.6, 6.6, 5.6, 8.0)
        maze.add_tri(6.25, 6.25, 6.5, 6.25, 5.6, 7.9)

        print('adding the survivor')
        maze.add_surv(9.61, 6.8)

        print('drawing the maze...')
        draw_maze(maze, swarm, source=source2)

    # run the simulation
    num_step = int(1000000)
    draw_maze(maze, swarm, source=source)
    plt.show()
    for frame in range(num_step):
        if frame % 1000 == 0 and frame != 0:
            print('{0} seconds'.format(int(frame/100)))
        
        '''
        fig, ax = plt.subplots(figsize=(10, 10))
        # ax.set_xticks(np.arange(0, width, 0.5))
        # ax.set_yticks(np.arange(0, height, 0.5))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        # ax.grid()
        ax.set_aspect(1)
        draw_maze(maze, swarm, source=source)
        plt.savefig(f"{frame_dir}/frame_{frame:06d}.png")
        plt.close(fig)
        '''

        found = swarm.rand_step_update(maze)
        # print(frame)
        if found:
            print('survivor found')
            for i in range(200): # add some stationary frames at the end
                fig, ax = plt.subplots(figsize=(10, 10))
                # ax.set_xticks(np.arange(0, width, 0.5))
                # ax.set_yticks(np.arange(0, height, 0.5))
                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
                # ax.grid()
                ax.set_aspect(1)

                draw_maze(maze, swarm, source=source)
                plt.savefig(f"{frame_dir}/frame_{frame:06d}.png")
                plt.close(fig)
                frame += 1
            break

        '''
        draw_maze(maze, swarm, source=source)
        plt.savefig(f"{frame_dir}/frame_{frame:06d}.png")
        plt.close(fig)
        '''

    print('# activated at least once: ', swarm.count_first_activated())
    print('# crashed: ', swarm.count_crashed())
    print('for c = 0.2, ct/4 = ',  0.2*swarm.t/4)

