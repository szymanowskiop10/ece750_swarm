from typing import Dict, List, Iterator, Tuple, TypeVar
from collections import deque
import heapq
import numpy as np

# Assumption: Less than MAX_NUM robots
MAX_NUM = 20000
ROBOT_RADIUS = 0.1
SENSORRANGE = 0.65

Location = TypeVar('Location')
T = TypeVar('T')
PointLocation = Tuple[float, float]
OccupyStatus = List[float]
RectLocation = Tuple[PointLocation, PointLocation] # The location of a rectangular obstacle is defined by its two corners
CircleLocation = Tuple[PointLocation, float]
TriLocation = Tuple[PointLocation, PointLocation, PointLocation] # 3 points determining a triangle
GridLocation = Tuple[int, int]
# ---------- Functions ----------
def point_line_dist(p_l1: PointLocation, p_l2: PointLocation, p_out: Tuple) -> float:
    p1, p2, p3 = np.array(p_l1), np.array(p_l2), np.array(p_out)
    if np.linalg.norm(p2-p1) < 0.001:
        print('invalid points on a line')
        return -1.0
    else:
        return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
    
def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def point_segment_dist(ver_1: PointLocation, ver_2: PointLocation, p_out: Tuple) -> float:
    v1_a = np.array(ver_2) - np.array(ver_1)
    v1_b = np.array(p_out) - np.array(ver_1)
    v2_a = np.array(ver_1) - np.array(ver_2)
    v2_b = np.array(p_out) - np.array(ver_2)
    if np.dot(v1_a, v1_b) < -0.001:
        return np.linalg.norm(v1_b)
    elif np.dot(v2_a, v2_b) < -0.001:
        return np.linalg.norm(v2_b)
    else:
        return point_line_dist(ver_1, ver_2, p_out)

def tri_area(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)
 
def in_tri_margin(x1: float, y1: float, x2: float, y2: float, 
                     x3: float, y3: float, x: float, y: float):
    margin = ROBOT_RADIUS
    A = tri_area(x1, y1, x2, y2, x3, y3)
    A1 = tri_area(x, y, x2, y2, x3, y3)
    A2 = tri_area(x1, y1, x, y, x3, y3)
    A3 = tri_area(x1, y1, x2, y2, x, y)
    if abs(A - A1 - A2 - A3) < 0.001:
        return True
    else:
        return point_segment_dist((x1,y1), (x2,y2), (x,y)) < margin \
        or point_segment_dist((x1,y1), (x3,y3), (x,y)) < margin \
        or point_segment_dist((x2,y2), (x3,y3), (x,y)) < margin
    
# ---------- Classes ----------
# Continuous Graph
class RealGraph:
    def __init__(self, width: float, height: float, grid_length: float):
        self.width = width
        self.height = height
        self.grid_length = grid_length
        self.rectangles: List[RectLocation] = []
        self.circles: List[CircleLocation] = []
        self.triangles: List[TriLocation] = []
        self.points: List[PointLocation] = []

    def add_cir(self, x: float, y: float, r: float):
        if x-r < x+r and y-r < y+r:
            self.circles.append(((x, y), r))
    
    def add_tri(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
        self.triangles.append(((x1, y1), (x2, y2), (x3, y3)))
    
# Discrete graph
class SquareGrid:
    def __init__(self, width: int, height: int, grid_length: float):
        self.width = width
        self.height = height
        self.grid_length = grid_length
        self.walls: List[GridLocation] = []
        self.points: List[Tuple[int, int]] = []
    
    def in_bounds(self, id: GridLocation):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        return id not in self.walls
    
    def four_neighbors(self, id):
        (x, y) = id
        four_neighbors = [(x-1, y), (x, y+1), (x+1, y), (x, y-1)]
        # if (x + y) % 2 == 0: four_neighbors.reverse()
        temp1, temp2 = [], []
        filtered = filter(self.in_bounds, four_neighbors)
        for point in filtered:
            temp1.append(point)
        filtered = filter(self.passable, temp1)
        for point in filtered:
            temp2.append(point)
        return temp2

    def twelve_neighbors(self, id):
        (x, y) = id
        twelve_neighbors = [(x, y+2),
                (x-1, y+1), (x, y+1), (x+1, y+1),
            (x-2, y), (x-1, y), (x+1, y), (x+2, y),
                (x-1, y-1), (x, y-1), (x+1, y-1),
                            (x, y-2)]
        temp1, temp2 = [], []
        filtered = filter(self.in_bounds, twelve_neighbors)
        for point in filtered:
            temp1.append(point)
        filtered = filter(self.passable, temp1)
        for point in filtered:
            temp2.append(point)
        return temp2

class GridWithMark(SquareGrid):
    def __init__(self, width: int, height: int, grid_length: float):
        super().__init__(width, height, grid_length)
        self.marks: Dict[GridLocation, OccupyStatus] = {}
        for i in range(height):
            for j in range(width):
                self.marks[(j, i)] = [0, 0]
    
    def remove_id(self, from_node: GridLocation, id: int) -> int:
        if self.in_bounds((from_node)):
            from_status = self.marks.get(from_node)
            if from_status[0] == id:
                from_status[0] = 0
            elif from_status[1] == id:
                from_status[1] = 0
            return 1
        else:
            return 0

    def add_id(self, to_node: GridLocation, id: int, settled: bool) -> int:
        to_status = self.marks.get(to_node)
        if to_status == None:
            print('out of map, crashing the robot', to_node)
            return 0
        to_val = id if not settled else id + MAX_NUM
        if sum(vertex > 0 for vertex in to_status) >= 2:
            print('vertex full, deleting robot no.{0}'.format(id))
            return 0
        elif to_status[0] == 0:
            to_status[0] = to_val
            self.marks[to_node] = to_status
            return 1
        elif to_status[1] == 0:
            to_status[1] = to_val
            self.marks[to_node] = to_status
            return 1

    def add_cir(self, x: float, y: float, r: float):
        left = int((x-r) // self.grid_length)
        right = int((x+r) // self.grid_length)
        bottom = int((y-r) // self.grid_length)
        up = int((y+r) // self.grid_length) 
        left = max(0, left)
        bottom = max(0, bottom)
        right = min(self.width, right+1)
        up = min(self.height, up+1)
        r_margin = r + ROBOT_RADIUS

        for i in range(left, right):
            for h in range(bottom, up):
                if ((x - self.grid_length*(i+0.5))**2 + (y - self.grid_length*(h+0.5))**2) < r_margin**2:
                    self.walls.append((i, h))

    def add_tri(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
        left = int(min(x1, x2, x3) // self.grid_length)
        right = int(max(x1, x2, x3) // self.grid_length)
        bottom = int(min(y1, y2, y3) // self.grid_length)
        up = int(max(y1, y2, y3) // self.grid_length)
        left = max(0, left)
        bottom = max(0, bottom)
        right = min(self.width, right+1)
        up = min(self.height, up+1)

        for i in range(left, right):
            for h in range(bottom, up):
                if in_tri_margin(x1, y1, x2, y2, x3, y3, self.grid_length*(i+0.5), self.grid_length*(h+0.5)):
                    self.walls.append((i, h))

# The main representation f the world
class Maze:
    def __init__(self, height: float, width: float, grid_length: float = 0.5):
        self.height = height
        self.width = width
        self.grid_length = grid_length
        self.real_map = RealGraph(width, height, grid_length)
        self.grids = GridWithMark(int(width//grid_length), int(height//grid_length), grid_length)
        self.survivors = []
    
    def add_rect(self, x1: float, y1: float, x2: float, y2: float):
        self.real_map.add_tri(x1, y1, x2, y2, x1, y2)
        self.real_map.add_tri(x1, y1, x2, y2, x2, y1)
        self.grids.add_tri(x1, y1, x2, y2, x1, y2)
        self.grids.add_tri(x1, y1, x2, y2, x2, y1)

    def add_cir(self, x: float, y: float, r: float):
        self.real_map.add_cir(x, y, r)
        self.grids.add_cir(x, y, r)

    def add_tri(self, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float):
        self.real_map.add_tri(x1, y1, x2, y2, x3, y3)
        self.grids.add_tri(x1, y1, x2, y2, x3, y3)

    def add_surv(self, x: float, y: float):
        self.survivors.append((x, y))

    def get_vertex(self, v_x, v_y):
        # get vertex valuex by coordinates
        return self.grids.marks.get((v_x, v_y))
    
    def get_height(self):
        return self.height
    
    def get_width(self):
        return self.width
    
    def get_cirs(self):
        return self.real_map.circles
    
    def get_tris(self):
        return self.real_map.triangles
    
    def get_walls(self):
        return self.grids.walls
    
    def get_people(self):
        return self.survivors

    def mark_robot(self, robot):
        # the discrete grids mark the robot whenever it submit the request
        # settled robot will be marked as MAX_NUM + index
        robot_stat = robot.get_status()
        robot_id = robot.get_index()

        prec = 4 if robot_stat == 3 else 1
        is_settled = True if robot_stat == 2 else False
        is_crashed = True if robot_stat == -1 else False

        temp_curr_loc = np.round(np.round(robot.get_location(), prec) // self.grid_length)
        temp_prev_loc = np.round(np.round(robot.get_prev_location(), prec) // self.grid_length)
        curr_loc = (temp_curr_loc[0], temp_curr_loc[1])
        prev_loc = (temp_prev_loc[0], temp_prev_loc[1])
        self.grids.remove_id(prev_loc, robot_id)
        if not is_crashed:
            add_result = self.grids.add_id(curr_loc, robot_id, is_settled)
            if not add_result:
                robot.crash(self)

    def robot_get_marked_id(self, robot, dir=-1):
        # return the neighbor's id that is marked by this settled robot
        if dir == -1:
            dir = robot.get_direction()
        loc = robot.get_location()
        vertex_loc = np.round(loc // self.grid_length)
        if dir == 0:
            vertex_loc[0] -= 1
        elif dir == 1:
            vertex_loc[1] -= 1
        elif dir == 2:
            vertex_loc[0] += 1
        else:
            vertex_loc[1] += 1
        vertex_id = self.grids.marks.get((vertex_loc[0], vertex_loc[1]))
        return max(vertex_id) - MAX_NUM 

    def robot_inquiry_surv(self, robot) -> bool:
        loc = robot.get_location()
        x_r, y_r = loc[0], loc[1]
        range = robot.get_sensor_range()
        for survivor in self.survivors:
            x_s, y_s = survivor[0], survivor[1]
            # print(loc)
            # print((x_s - x_r) ** 2 + (y_s - y_r) ** 2)
            # print(range ** 2)
            if (x_s - x_r) ** 2 + (y_s - y_r) ** 2 < range ** 2:
                return True
        return False

    def robot_inquiry_general(self, robot, swarm):
        # return the status of the nearby 12 vertices in discrete representation
        is_wall = [True]*12
        neighbor_count = np.zeros(12) # num of robot in each neighbor_dir
        neighbor_dir = np.full(12,-1) # the neighbor_dir marked by these robots
        loc = np.round(np.array(robot.get_location()) // self.grid_length)
        x, y = loc[0], loc[1]

        passable_points = self.grids.twelve_neighbors((x,y))
        # print('passable_points: ', passable_points)
        twelve_neighbors = [(x, y+2),
                (x-1, y+1), (x, y+1), (x+1, y+1),
            (x-2, y), (x-1, y), (x+1, y), (x+2, y),
                (x-1, y-1), (x, y-1), (x+1, y-1),
                            (x, y-2)]
        for i in range(12):
            if twelve_neighbors[i] in passable_points:
                is_wall[i] = False

        # only get mark direction when the grid has 1 robot
        # check left
        if (x-2, y) in passable_points:
            vertex_id = self.grids.marks.get((x-2, y))
            count = sum(id > 0 for id in vertex_id)
            if count == 1:
                neighbor_id = max(vertex_id)
                if neighbor_id > MAX_NUM: 
                    neighbor_dir[4] = swarm.get_robot_dir(neighbor_id-MAX_NUM)
                neighbor_count[4] = 1
            elif count == 2:
                neighbor_count[4] = 2
        if (x-1, y) in passable_points:
            vertex_id = self.grids.marks.get((x-1, y))
            count = sum(id > 0 for id in vertex_id)
            if count == 1:
                neighbor_id = max(vertex_id)
                if neighbor_id > MAX_NUM: 
                    neighbor_dir[5] = swarm.get_robot_dir(neighbor_id-MAX_NUM)
                neighbor_count[5] = 1
            elif count == 2:
                neighbor_count[5] = 2 
        
        # check down
        if (x, y-2) in passable_points:
            vertex_id = self.grids.marks.get((x, y-2))
            count = sum(id > 0 for id in vertex_id)
            if count == 1:
                neighbor_id = max(vertex_id)
                if neighbor_id > MAX_NUM:
                    neighbor_dir[11] = swarm.get_robot_dir(neighbor_id-MAX_NUM)
                neighbor_count[11] = 1
            elif count == 2:
                neighbor_count[11] = 2
        if (x, y-1) in passable_points:
            vertex_id = self.grids.marks.get((x, y-1))
            count = sum(id > 0 for id in vertex_id)
            if count == 1:
                neighbor_id = max(vertex_id)
                if neighbor_id > MAX_NUM:
                    neighbor_dir[9] = swarm.get_robot_dir(neighbor_id-MAX_NUM)
                neighbor_count[9] = 1
            elif count == 2:
                neighbor_count[9] = 2

        # check right
        if (x+2, y) in passable_points:
            vertex_id = self.grids.marks.get((x+2, y))
            count = sum(id > 0 for id in vertex_id)
            if count == 1:
                neighbor_id = max(vertex_id)
                if neighbor_id > MAX_NUM:
                    neighbor_dir[7] = swarm.get_robot_dir(neighbor_id-MAX_NUM)
                neighbor_count[7] = 1
            elif count == 2:
                neighbor_count[7] = 2
        if (x+1, y) in passable_points:
            vertex_id = self.grids.marks.get((x+1, y))
            count = sum(id > 0 for id in vertex_id)
            if count == 1:
                neighbor_id = max(vertex_id)
                if neighbor_id > MAX_NUM:
                    neighbor_dir[6] = swarm.get_robot_dir(neighbor_id-MAX_NUM)
                neighbor_count[6] = 1
            elif count == 2:
                neighbor_count[6] = 2

        # check up
        if (x, y+2) in passable_points:
            vertex_id = self.grids.marks.get((x, y+2))
            count = sum(id > 0 for id in vertex_id)
            if count == 1:
                neighbor_id = max(vertex_id)
                if neighbor_id > MAX_NUM:
                    neighbor_dir[0] = swarm.get_robot_dir(neighbor_id-MAX_NUM)
                neighbor_count[0] = 1
            elif count == 2:
                neighbor_count[0] = 2
        if (x, y+1) in passable_points:
            vertex_id = self.grids.marks.get((x, y+1))
            count = sum(id > 0 for id in vertex_id)
            if count == 1:
                neighbor_id = max(vertex_id)
                if neighbor_id > MAX_NUM:
                    neighbor_dir[2] = swarm.get_robot_dir(neighbor_id-MAX_NUM)
                neighbor_count[2] = 1
            elif count == 2:
                neighbor_count[2] = 2

        return is_wall, neighbor_count, neighbor_dir
