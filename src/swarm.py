from typing import Dict, List, Iterator, Tuple, TypeVar
import math
import random
import copy
import numpy as np
from maze import Maze, unit_vector, MAX_NUM, ROBOT_RADIUS, SENSORRANGE

# Maze and Swarm keep global information, but 
#   every MobileRobot instance can only access its local information
# Robot only records the continous coordinates

class MobileRobot:
    def __init__(self, index: int = 1, location: List[float] = [.0,.0], 
                 source: List[float] = [1.0,1.0], status: int = 0
                 , grid_length: float = 0.5, step_length: float = 0.01):
        self.index = index # 1 <= index < MAX_NUM
        self.location = np.array(location)
        self.prev_location = np.array(location)
        self.status = status
        self.first_activated = False
        self.source = np.array(source)
        self.radius = 0.05
        self.grid_length = grid_length
        self.step_length = step_length
        self.speed = 1.0
        self.move_vector = np.array([1.0, .0]) # always a unit vector
        self.move_target = np.array([.0, .0])
        self.settled_after_moving = False
        self.direction = -1 # To which neighbor_dir it marks
        self.planned_direction = -1
        self.find_surv = False
        self.next_in_path = -1
        self.sensor_range = SENSORRANGE
        self.c = .0 # crash rate
    
    # status 0: the robot is not activated
    # status 1: the robot is activated and at rest
    # status 2: the robot is in settled state
    # statue 3: the robot is moving
    # status -1: the robot is crashed
        
    # neighbor_dir 0, 1, 2, 3: left, down, right, up
    def get_location(self):
        return self.location
    
    def get_prev_location(self):
        return self.prev_location
    
    def get_radius(self):
        return self.radius
    
    def get_status(self):
        return self.status
    
    def get_index(self):
        return self.index
    
    def get_direction(self):
        return self.direction
    
    def get_activated_once(self):
        return self.first_activated
    
    def get_sensor_range(self):
        return self.sensor_range
    
    def get_next_in_path(self):
        return self.next_in_path
    
    def upload_maze(self, maze):
        # Upload its status and location to the maze at the end of each step
        maze.mark_robot(self)
        return 0
    
    def is_source_open(self, maze, x, y):
        ver_s = maze.get_vertex(x, y)
        if sum(item > 0 for item in ver_s) < 2:
            return True
        return False
        
    def receive_surv_info(self, last_dir: int, maze, swarm):
        self.find_surv = True
        self.next_in_path = (last_dir + 2) % 4
        if np.linalg.norm(self.location - self.source) < 0.001:
            print('info has reached the source')
            return 1
        else:
            return self.send_surv_info(maze, swarm)

    def send_surv_info(self, maze, swarm):
        next_id = maze.robot_get_marked_id(self)
        return swarm.robot_list[next_id-1].receive_surv_info(self.direction, maze, swarm)

    def search_surv(self, maze, swarm):
        if self.status != 2:
            return 0
        elif maze.robot_inquiry_surv(self):
            # print('survivor found, start propogating')
            self.find_surv = True
            return self.send_surv_info(maze, swarm)
        return 0
    
    def activate(self, maze) -> int:
        global activation_count
        if self.status == 0:
            self.status = 1
            if not self.first_activated:
                s_x = int(self.source[0] // self.grid_length)
                s_y = int(self.source[1] // self.grid_length)
                ver_source = maze.get_vertex(s_x, s_y)
                source_count = sum(id > 0 for id in ver_source)
                if source_count < 2:
                    self.location = copy.deepcopy(self.source)
                    self.first_activated = True
                    if source_count == 0:
                        self.status = 2 # first robot entering the maze
                        self.upload_maze(maze)
                        return self.index
                    self.upload_maze(maze)
                else:
                    self.status = 0 # source is filled, cannot insert now
                    return 0
        return 0
    
    def crash(self, maze: Maze):
        if self.status != 0 and self.status != 2:
            print('robot {0} has crashed'.format(self.index))
            self.status = -1
            self.direction = -1
            self.prev_location = self.location
            self.upload_maze(maze)

    def crash_with_prob(self, maze: Maze):
        if self.c > .002 and random.random() < self.c:
            self.crash(maze)
        
    def deactivate(self):
        if self.status == 1 or self.status == 3:
            self.status = 0
    
    def cont_move(self, maze: Maze, swarm) -> int:
        if self.status == 2:
            return 0
        elif self.status == 0 or self.status == -1:
            return 2 # not activated yet or already crashed
        elif self.status == 3:
            self.prev_location = copy.deepcopy(self.location)
            self.location += self.move_vector * self.speed * self.step_length
            if np.linalg.norm(self.move_target - self.location) < 0.001:
                if not self.settled_after_moving:
                    self.deactivate() # move complete
                else:
                    self.status = 2 # move complete and settled
                    self.direction = self.planned_direction   
            maze.mark_robot(self)
            return 1
        else:
            is_wall, neighbor_count, neighbor_dir = maze.robot_inquiry_general(self, swarm)

        # check settled neighbor (1 grid away)
        if neighbor_dir[5] == 2:
            self.move_vector = np.array([-1.0, .0])
            self.move_target = self.location + self.move_vector * self.grid_length 
            self.status = 3
            return 1
        elif neighbor_dir[9] == 3:
            self.move_vector = np.array([.0, -1.0])
            self.move_target = self.location + self.move_vector * self.grid_length 
            self.status = 3
            return 1
        elif neighbor_dir[6] == 0:
            self.move_vector = np.array([1.0, .0])
            self.move_target = self.location + self.move_vector * self.grid_length 
            self.status = 3
            return 1
        elif neighbor_dir[2] == 1:
            self.move_vector = np.array([.0, 1.0])
            self.move_target = self.location + self.move_vector * self.grid_length 
            self.status = 3
            return 1
        
        # check empty point (1 grid away)
        if not is_wall[5] and neighbor_count[5] == 0 and neighbor_count[4] == 0: 
            self.move_vector = np.array([-1.0, .0])
            self.move_target = self.location + self.move_vector * self.grid_length 
            self.status = 3
            self.settled_after_moving = True
            self.planned_direction = 2
            return 1
        elif not is_wall[9] and neighbor_count[9] == 0 and neighbor_count[11] == 0:
            self.move_vector = np.array([.0, -1.0])
            self.move_target = self.location + self.move_vector * self.grid_length 
            self.status = 3
            self.settled_after_moving = True
            self.planned_direction = 3
            return 1
        elif not is_wall[6] and neighbor_count[6] == 0 and neighbor_count[7] == 0:
            self.move_vector = np.array([1.0, .0])
            self.move_target = self.location + self.move_vector * self.grid_length 
            self.status = 3
            self.settled_after_moving = True
            self.planned_direction = 0
            return 1
        elif not is_wall[2] and neighbor_count[2] == 0 and neighbor_count[0] == 0:
            self.move_vector = np.array([.0, 1.0])
            self.move_target = self.location + self.move_vector * self.grid_length 
            self.status = 3
            self.settled_after_moving = True
            self.planned_direction = 1
            return 1
        return 2

class Swarm:
    def __init__(self, step_length: float = 0.01,
                 t: float = 0.0):
        self.robot_list = [] # swarm id starts from 1
        self.survivor_found = False
        self.last_has_entered = 0
        self.step_length = step_length
        self.t = t
        self.step_count = 0
        self.source_id = -1
        self.step_per_crash = int(30.0/self.step_length)

    def get_num(self) -> int:
        return len(self.robot_list)

    def get_activated_once(self, id: int) -> bool:
        return self.robot_list[id-1].get_activated_once()
    
    def get_geometry(self, id: int) -> List:
        return (self.robot_list[id-1].get_location(), self.robot_list[id-1].get_radius())

    def get_robot_dir(self, id: int) -> int:
        return self.robot_list[id-1].get_direction()
    
    def get_robot_loc(self, id: int) -> int:
        return self.robot_list[id-1].get_location()
    
    def add_robot(self, robot: MobileRobot) -> int:
        if self.survivor_found:
            return -1
        else:
            self.robot_list.append(robot)
            return 1
        
    def add_robot_batch(self, num_robot: int, maze_source: List[float]):
        if num_robot > MAX_NUM:
            print('cannot add, too many robots')
        else:
            for i in range(num_robot):
                robot_id = i+1
                self.add_robot(MobileRobot(index=robot_id, location=[-1, -1], 
                                           source= maze_source, status=0, step_length=self.step_length))
        
    def rand_step_update(self, maze: Maze):
        if self.survivor_found:
            return 1
        else:
            self.t += self.step_length
            self.step_count += 1
            self.rand_activation(maze)
            for robot in self.robot_list:
                if self.step_count % self.step_per_crash == 0:
                    robot.crash_with_prob(maze)
                robot.cont_move(maze, self)
                result = robot.search_surv(maze, self)
                if result:
                    self.survivor_found = True
                    print('dispersion ends at {0} s'.format(self.t))
                    return 1
            return 0
                         
    def rand_activation(self, maze, rate=1, ind_priority=1):
        # rate: lambda
        # step_length: the smallest time step in simulation
        beta = 1.0/rate
        num_robot = len(self.robot_list)
        rv_list = np.random.default_rng().exponential(scale=beta, size=num_robot)
        activation_id = np.array([rv < self.step_length for rv in rv_list])
        if not ind_priority:
            for i in range(num_robot):
                if activation_id[i]:
                    self.robot_list[i].activate(maze)
        else:
            for i in range(num_robot):
                if activation_id[i]:
                    if self.robot_list[i].get_activated_once():
                        self.robot_list[i].activate(maze)
                    elif self.last_has_entered == i:
                        id = self.robot_list[i].activate(maze)
                        if id != 0:
                            self.source_id = id
                        self.last_has_entered += 1

    def get_path_to_surv(self, maze) -> List:
        path = []
        if self.survivor_found: 
            id = self.source_id
            next_in_path = self.robot_list[id-1].get_next_in_path()
            path.append(self.robot_list[id-1].get_location())
            while next_in_path != -1:
                id = maze.robot_get_marked_id(self.robot_list[id-1], next_in_path)
                path.append(self.robot_list[id-1].get_location())
                next_in_path = self.robot_list[id-1].get_next_in_path()
        # print('path: ', path)
        return path

    def count_first_activated(self):
        count = 0
        for robot in self.robot_list:
            if robot.get_activated_once():
                count += 1
        return count
    
    def count_crashed(self):
        count = 0
        for robot in self.robot_list:
            if robot.get_status() == -1:
                count += 1
        return count


        




        


    
    
        
