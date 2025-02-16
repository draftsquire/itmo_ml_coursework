import numpy as np
from typing import List, Tuple, Optional, Literal
import matplotlib.pyplot as plt
from mecanum_gen import obstacle_positions_square
from heapq import heappush, heappop
import random
import trimesh
from os.path import join
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely.affinity import translate

class Node:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.g = float('inf')  # Cost from start (for A*)
        self.h = 0.0  # Heuristic to goal (for A*)
        self.f = float('inf')  # Total cost (for A*)
        self.parent = None
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6

class PathPlanner:
    def __init__(self, 
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 obstacles: List[Tuple[float, float, float]],
                 bounds: Tuple[float, float, float, float],
                 algorithm: Literal['astar', 'rrt'] = 'astar',
                 grid_resolution: float = 0.1,
                 rrt_step_size: float = 0.5,
                 max_iterations: int = 5000,
                 robot_radius: float = 0.1,  # Robot's radius
                 robot_footprint: Optional[List[Tuple[float, float]]] = None):  # Custom robot shape
        """
        Initialize path planner with robot size consideration
        
        Args:
            start: Starting position (x, y)
            goal: Goal position (x, y)
            obstacles: List of circular obstacles (x, y, radius)
            bounds: Environment bounds (min_x, max_x, min_y, max_y)
            algorithm: 'astar' or 'rrt'
            grid_resolution: Size of grid cells (for A*)
            rrt_step_size: Step size for RRT extension
            max_iterations: Maximum number of iterations
            robot_radius: Radius of the robot (if circular)
            robot_footprint: List of points defining robot's shape polygon (if not circular)
        """
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.bounds = bounds  # Keep original bounds for visualization
        self.algorithm = algorithm
        self.resolution = grid_resolution
        self.rrt_step_size = rrt_step_size
        self.max_iterations = max_iterations
        self.robot_radius = robot_radius
        
        # Create robot footprint
        if robot_footprint is None:
            self.robot_footprint = Point(0, 0).buffer(robot_radius)
        else:
            self.robot_footprint = Polygon(robot_footprint)
        
        # Process obstacles
        self.obstacles = self._process_obstacles(obstacles)
        self.obstacle_polygons = self._create_obstacle_polygons()
        self.merged_obstacles = self._merge_close_obstacles()
        
        # Calculate grid size based on actual environment size
        self.x_size = int(np.ceil((bounds[1] - bounds[0]) / grid_resolution))
        self.y_size = int(np.ceil((bounds[3] - bounds[2]) / grid_resolution))
        self.grid = np.zeros((self.x_size, self.y_size))
        
        # For visualization
        self.visited_nodes = []
        self.rrt_nodes = []
        
        # Motion model for A*
        self.motions = [
            (1, 0), (0, 1), (-1, 0), (0, -1),  # 4-connected
            (1, 1), (-1, 1), (-1, -1), (1, -1)  # diagonals
        ]

    def _process_obstacles(self, obstacles):
        """Process obstacles for 2D path planning"""
        processed_obstacles = []
        
        # Process the passed obstacles directly instead of using obstacle_positions_square
        for x, y, radius in obstacles:
            processed_obstacles.append({
                'name': 'obstacle',
                'position': (x, y),
                'radius': radius,
                'type': 'circle'
            })
        
        return processed_obstacles

    def _create_obstacle_polygons(self):
        """Convert obstacles to Shapely polygons and dilate by robot size"""
        polygons = []
        for obs in self.obstacles:
            if obs['type'] == 'polygon':
                # Create polygon and dilate by robot size
                poly = Polygon(obs['hull_vertices'])
                dilated_poly = poly.buffer(self.robot_radius)
                polygons.append(dilated_poly)
            else:
                # Create circle and dilate by robot size
                circle = Point(obs['position']).buffer(obs['radius'] + self.robot_radius)
                polygons.append(circle)
        return polygons

    def _merge_close_obstacles(self):
        """Merge obstacles that are close to each other"""
        return unary_union(self.obstacle_polygons)

    def _translate_polygon(self, polygon: Polygon, x: float, y: float) -> Polygon:
        """Translate a polygon by given x, y offsets"""
        coords = list(polygon.exterior.coords)
        new_coords = [(px + x, py + y) for px, py in coords]
        return Polygon(new_coords)

    def _is_collision(self, x: float, y: float) -> bool:
        """Check collision considering robot's shape"""
        # Create robot polygon at the given position
        robot_at_pos = self._translate_polygon(self.robot_footprint, x, y)
        
        # Check bounds
        if not (self.bounds[0] <= x <= self.bounds[1] and 
                self.bounds[2] <= y <= self.bounds[3]):
            return True
        
        # Check collision with obstacles
        return self.merged_obstacles.intersects(robot_at_pos)

    def _is_path_collision(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if path is collision-free considering robot's shape"""
        # Create a line for the path
        path = LineString([(x1, y1), (x2, y2)])
        
        # Create a buffer around the path based on robot size
        path_buffer = path.buffer(self.robot_radius)
        
        # Check collision with obstacles
        return self.merged_obstacles.intersects(path_buffer)

    def _plan_astar(self) -> Optional[List[Tuple[float, float]]]:
        """A* path planning with corrected coordinate transformation"""
        # Transform start and goal to grid coordinates
        start_grid_x = int((self.start.x - self.bounds[0]) / self.resolution)
        start_grid_y = int((self.start.y - self.bounds[2]) / self.resolution)
        goal_grid_x = int((self.goal.x - self.bounds[0]) / self.resolution)
        goal_grid_y = int((self.goal.y - self.bounds[2]) / self.resolution)
        
        self.start.g = 0
        self.start.h = np.hypot(self.start.x - self.goal.x, self.start.y - self.goal.y)
        self.start.f = self.start.g + self.start.h
        
        open_set = []
        heappush(open_set, self.start)
        closed_set = set()
        
        while open_set and len(self.visited_nodes) < self.max_iterations:
            current = heappop(open_set)
            self.visited_nodes.append((current.x, current.y))
            
            if np.hypot(current.x - self.goal.x, current.y - self.goal.y) < self.resolution:
                path = []
                while current is not None:
                    path.append((current.x, current.y))
                    current = current.parent
                return path[::-1]
            
            closed_set.add((current.x, current.y))
            
            for dx, dy in self.motions:
                x = current.x + dx * self.resolution
                y = current.y + dy * self.resolution
                
                if self._is_collision(x, y):
                    continue
                
                neighbor = Node(x, y)
                if (neighbor.x, neighbor.y) in closed_set:
                    continue
                
                tentative_g = current.g + np.hypot(dx * self.resolution, dy * self.resolution)
                
                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = np.hypot(neighbor.x - self.goal.x, neighbor.y - self.goal.y)
                    neighbor.f = neighbor.g + neighbor.h
                    
                    if neighbor not in open_set:
                        heappush(open_set, neighbor)
        
        return None

    def _plan_rrt(self) -> Optional[List[Tuple[float, float]]]:
        """RRT path planning"""
        self.rrt_nodes = [self.start]
        
        for _ in range(self.max_iterations):
            # Random sampling
            if random.random() < 0.05:  # 5% chance to sample goal
                x_rand, y_rand = self.goal.x, self.goal.y
            else:
                x_rand = random.uniform(self.bounds[0], self.bounds[1])
                y_rand = random.uniform(self.bounds[2], self.bounds[3])
            
            # Find nearest node
            nearest = min(self.rrt_nodes, 
                        key=lambda n: np.hypot(n.x - x_rand, n.y - y_rand))
            
            # Extend towards random point
            theta = np.arctan2(y_rand - nearest.y, x_rand - nearest.x)
            x_new = nearest.x + self.rrt_step_size * np.cos(theta)
            y_new = nearest.y + self.rrt_step_size * np.sin(theta)
            
            if self._is_collision(x_new, y_new):
                continue
            
            if self._is_path_collision(nearest.x, nearest.y, x_new, y_new):
                continue
            
            new_node = Node(x_new, y_new)
            new_node.parent = nearest
            self.rrt_nodes.append(new_node)
            
            # Check if goal reached
            if np.hypot(x_new - self.goal.x, y_new - self.goal.y) < self.rrt_step_size:
                final_node = Node(self.goal.x, self.goal.y)
                final_node.parent = new_node
                self.rrt_nodes.append(final_node)
                
                # Extract path
                path = []
                node = final_node
                while node is not None:
                    path.append((node.x, node.y))
                    node = node.parent
                return path[::-1]
        
        return None

    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """Plan path using selected algorithm"""
        if self.algorithm == 'astar':
            return self._plan_astar()
        else:
            return self._plan_rrt()

    def visualize(self, path: Optional[List[Tuple[float, float]]] = None):
        """Enhanced visualization with actual obstacle geometry"""
        plt.figure(figsize=(10, 10))
        
        # Plot bounds
        plt.plot([self.bounds[0], self.bounds[1]], [self.bounds[2], self.bounds[2]], 'k-', label='Environment')
        plt.plot([self.bounds[0], self.bounds[1]], [self.bounds[3], self.bounds[3]], 'k-')
        plt.plot([self.bounds[0], self.bounds[0]], [self.bounds[2], self.bounds[3]], 'k-')
        plt.plot([self.bounds[1], self.bounds[1]], [self.bounds[2], self.bounds[3]], 'k-')
        
        # Plot actual obstacle geometry
        from mecanum_gen import obstacle_positions_square
        
        # Plot boxes
        for i in range(9):  # boxx0 to boxx8
            name = f'boxx{i}.stl'
            if name in obstacle_positions_square:
                pos = obstacle_positions_square[name]
                box_size = 0.4  # Actual box size
                rect = plt.Rectangle(
                    (pos[0] - box_size/2, pos[1] - box_size/2),
                    box_size, box_size,
                    color='red', alpha=0.3
                )
                plt.gca().add_patch(rect)
        
        # Plot borders
        for i in range(4):  # border0 to border3
            name = f'border{i}.stl'
            if name in obstacle_positions_square:
                pos = obstacle_positions_square[name]
                border_width = 0.2  # Actual border width
                if i < 2:  # Horizontal borders
                    rect = plt.Rectangle(
                        (pos[0] - 4, pos[1] - border_width/2),
                        8, border_width,
                        color='gray', alpha=0.3
                    )
                else:  # Vertical borders
                    rect = plt.Rectangle(
                        (pos[0] - border_width/2, pos[1] - 4),
                        border_width, 8,
                        color='gray', alpha=0.3
                    )
                plt.gca().add_patch(rect)
        
        # Plot robot footprint at start and goal
        start_robot = self._translate_polygon(self.robot_footprint, self.start.x, self.start.y)
        goal_robot = self._translate_polygon(self.robot_footprint, self.goal.x, self.goal.y)
        
        x, y = start_robot.exterior.xy
        plt.fill(x, y, 'g', alpha=0.5, label='Robot')
        plt.plot(x, y, 'g-', alpha=0.7)
        
        x, y = goal_robot.exterior.xy
        plt.fill(x, y, 'r', alpha=0.5)
        plt.plot(x, y, 'r-', alpha=0.7)
        
        # Plot path and robot footprint along path
        if path is not None:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label='Path')
            
            # Plot robot footprint at few points along the path
            for i in range(0, len(path), max(1, len(path)//10)):  # Show ~10 poses along path
                robot_at_point = self._translate_polygon(self.robot_footprint, path[i, 0], path[i, 1])
                x, y = robot_at_point.exterior.xy
                plt.plot(x, y, 'b--', alpha=0.2)
        
        plt.axis('equal')
        plt.grid(True)
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
                  fancybox=True, shadow=True, ncol=1)
        plt.title(f'Path Planning ({self.algorithm.upper()})')
        plt.tight_layout()
        plt.show() 