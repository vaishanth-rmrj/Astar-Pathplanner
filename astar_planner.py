import cv2
import matplotlib.pyplot as plt

from generate_map_utils import *

"""
Class to process map image

"""
class MazeMap:    

    def __init__(self, map_img, robot_radius):
        self.map_img = map_img
        self.obstacle_coordinates = self.obstacle_constructor()
        self.robot_radius = robot_radius

    def obstacle_constructor(self):
        """
        method to construct a boundary

        Returns:
            list: list of all obstacle co-ordinates
        """
        obstacle_coordinates = list()
        # constructing from map
        for row in range(self.map_img.shape[0]):
            for col in range(self.map_img.shape[1]):
                if np.array_equal(self.map_img[row, col], [47, 47, 211]) or np.array_equal(self.map_img[row, col], [154, 154, 239]):
                    obstacle_coordinates.append((row, col))

        return obstacle_coordinates        

    def check_if_obstacle(self, coordinate):
        """
        method to check if a coordinate is obstacle

        Args:
            coordinate (tuple): cordinate to check

        Returns:
            bool: True if detected
        """
        if coordinate[0] > self.map_img.shape[0] or coordinate[1] > self.map_img.shape[1]:
            return True

        for x_obs,y_obs in self.obstacle_coordinates:
            if(((coordinate[0] - x_obs)**2+(coordinate[1] - y_obs)**2)**0.5 < self.robot_radius):
                return True
        
        return False
        
class NodeCatlogue:   
    """
    class to keep track of all nodes created
    """ 

    def __init__(self, goal_coordinate):
        self.node_catalogue = list()
        self.goal_coordinate = goal_coordinate
        

    def create_node(self, coordinate, theta,  parent_node):
        """
        method to create a node

        Args:
            coordinate (tuple): coordinate of the node
            theta (float): orientation of the node
            parent_node (tuple): parent node with cost2come, cost2go

        Returns:
            list: [coordinate, theta, parent_coordinate, cost2come, cost2go, total_cost]
        """

        current_cost2come = self.calc_eucledian_dist(coordinate, parent_node[0])
        cost2go = self.calc_eucledian_dist(coordinate, self.goal_coordinate)
        parent_cost = parent_node[3]
        total_cost2come = current_cost2come + parent_cost

        total_cost = total_cost2come + cost2go
        total_theta = theta + parent_node[1]

        node = [coordinate, theta, parent_node[0], total_cost2come, cost2go, total_cost]
        self.node_catalogue.append(node)

        return node

    def calc_eucledian_dist(self, point1, point2):
        """
        method to calculate the eucledian distance

        Args:
            point1 (tuple): point 1 coordinates
            point2 (tuple): point 2 coordinates

        Returns:
            int: eucledian distance between point 1 and point 2
        """
        return int(((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5)

    def fetch_node(self, coordinate):
        """
        method to fetch the node give coordinate

        Args:
            coordinate (tuple): the node coordinate

        Returns:
            Node: requested node object
        """
        for node in self.node_catalogue:
            if coordinate[0] == node[0][0] and coordinate[1] == node[0][1]:
                return node
        return False

    def fetch_parent_node(self, current_node):
        """
        method to fetch the parent node for a given node

        Args:
            node (Node): Node object

        Returns:
            Node: parent node object
        """
        for node in self.node_catalogue:
            if current_node[2][0] == node[0][0] and current_node[2][1] == node[0][1]:
                return node

        return False

class AStarPlanner:
    """
    class to perform AStar planning
    """

    def __init__(self, start_coordinate, goal_coordinate, map_obj, start_orientation, goal_orientation, step_size):
        self.start_coordinate = start_coordinate
        self.goal_coordinate = goal_coordinate
        self.map = map_obj

        self.start_orientation = start_orientation
        self.goal_orientation = goal_orientation
        self.step_size = step_size
        self.check_threshold_radius = 5
        self.goal_threshold_radius = 20

        self.draw_start_goal_points()
        self.is_plan_feasible = self.check_start_goal_feasibility()

        self.open_node = []
        self.closed_node = []

        # action set
        self.action_set = [0, 30, 60, 90, 120] 
        self.node_catalogue = NodeCatlogue(self.goal_coordinate)

    def draw_start_goal_points(self):
        cv2.circle(self.map.map_img, (self.start_coordinate[1], self.start_coordinate[0]), 2, (85, 139, 47), 2)
        cv2.circle(self.map.map_img, (self.goal_coordinate[1], self.goal_coordinate[0]), 2, (85, 139, 47), 2)

    def check_start_goal_feasibility(self):
        """
        method to check if the start and goal node are not obstacle

        Returns:
            bool: False if not obstacle
        """
        if self.map.check_if_obstacle(self.start_coordinate) or self.map.check_if_obstacle(self.goal_coordinate):
            return False

        if self.start_coordinate > self.map.map_img.shape or self.goal_coordinate > self.map.map_img.shape:
            return False

        return True        

    def check_if_visited(self, coordinate):
        """
        method to check if the coordinate is already visited

        Args:
            coordinate (tuple): coordinate to check

        Returns:
            bool: False if not visited
        """
        for node in self.node_catalogue.node_catalogue:
            if(((node[0][0] - coordinate[0])**2 + (node[0][1] - coordinate[1])**2)**0.5 < self.check_threshold_radius):
                return True
        return False

    def calc_eucledian_dist(self, point1, point2):
        return int(((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5)

    def check_next_node_validity(self, current_node, next_node):
        """
        method to check if the node is reachable by verifying all the points connecting the two nodes

        Args:
            current_node (tuple): current node coordinate
            next_node (tuple): next node coordinate

        Returns:
            bool: returns true if in obstacle space
        """

        x1, y1 = current_node
        x2, y2 = next_node

        try:
            slope = (y2-y1)/(x2-x1)
        except:
            slope = 0

        intercept = y2-slope*x1
        for x in range(min(x1, x2), max(x1,x2)):
            y_pred = int(slope*x + intercept)            
            if self.map.check_if_obstacle((x, y_pred)):
                return True

        return False           


    def get_next_move(self, current_node):
        """
        method to get next move nodes

        Args:
            current_node (tuple): current coordinate

        Returns:
            list: list of next move nodes
        """
        next_nodes = [[(int(current_node[0][0] + self.step_size * np.cos(np.deg2rad(theta+current_node[1]-60))), int(current_node[0][1]+self.step_size*np.sin(np.deg2rad(theta+current_node[1]-60)))), theta+current_node[1]-60] for theta in self.action_set]

        valid_nodes = []
        for node in next_nodes:
            if not self.check_next_node_validity(current_node[0], node[0]):
                if not self.map.check_if_obstacle(node[0]):
                    valid_nodes.append(node)

        return valid_nodes

    def low_cost_node_index(self, open_nodes):
        """
        method to get the index of the low cost node

        Args:
            open_nodes (list): list of all the open nodes

        Returns:
            int: index of the element in the open node with low total cost
        """
        min_cost = float('inf')    
        min_cost_node = []
        open_nodes.reverse()
        for node in open_nodes:
            if node[4] < min_cost:
                min_cost = node[4]
                min_cost_node.clear()
                min_cost_node = node.copy()

        for index, node in enumerate(open_nodes):
            if node[0][0] == min_cost_node[0][0] and min_cost_node[0][1] == node[0][1]:
                return index        
        

    def search_for_path(self):
        """
        main method to search through the map

        Returns:
            list, int: list of the planned path, length of visited nodes
        """

        if self.is_plan_feasible:

            start_node = [self.start_coordinate, self.start_orientation, self.start_coordinate, 0, self.calc_eucledian_dist(self.start_coordinate, self.goal_coordinate), 0+ self.calc_eucledian_dist(self.start_coordinate, self.goal_coordinate)]
            self.open_node.append(start_node)# adding start node to the open queue

            iteration_count = 0
            while len(self.open_node):          

                current_node = self.open_node.pop(self.low_cost_node_index(self.open_node))
                self.closed_node.append(current_node)
                
                if self.calc_eucledian_dist(current_node[0], self.goal_coordinate) < self.goal_threshold_radius :
                    print("Hurray!! Puzzle has been solved")
                    break
                
                next_nodes_with_theta = self.get_next_move(current_node)

                for node_with_theta in next_nodes_with_theta: 
                    if not self.check_if_visited(node_with_theta[0]):

                        node_coordinate = node_with_theta[0]
                        node_theta = node_with_theta[1]
                        parent_node = current_node

                        # draw line connecting the node
                        cv2.line(self.map.map_img, (parent_node[0][1], parent_node[0][0]), (node_coordinate[1], node_coordinate[0]), (85, 139, 47), 1)
                        
                        # creating new node
                        new_node = self.node_catalogue.create_node(node_coordinate, node_theta, parent_node)
                        already_exist = False
                        for index, open_node in enumerate(self.open_node):
                            if new_node[0][0] == open_node[0][0] and new_node[0][1] == open_node[0][1] and new_node[5] < open_node[5]:
                                self.open_node.pop(index)
                                self.open_node.append(new_node)
                                already_exist = True    
                        
                        if not already_exist:
                            self.open_node.append(new_node)

                # drawing the open nodes
                for next_node in self.open_node:
                    cv2.circle(self.map.map_img, (next_node[0][1], next_node[0][0]), 1, (85, 139, 47), 1)                

                iteration_count += 1
                
                cv2.imshow('Frame', self.map.map_img)                
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break                

            cv2.destroyAllWindows()
            return self.plan_path(), self.map.map_img

        else:        
            print("Plan is not feasible because either start or goal coordinate is inside an obstacle")

         
    
    def plan_path(self):
        """
        method to back track through the nodes to get the optimal path

        Returns:
            list: list of planned path
        """

        planned_path = []              
        planned_path.append(self.goal_coordinate)

        for node in self.node_catalogue.node_catalogue:
            if(((node[0][0] - self.goal_coordinate[0])**2+(node[0][1] - self.goal_coordinate[1])**2)**0.5 < self.goal_threshold_radius):
                current_node = node
                break

        planned_path.append(current_node[0])       

        while True:
            current_node = self.node_catalogue.fetch_parent_node(current_node)
            planned_path.append(current_node[0])

            if current_node[2][0] == self.start_coordinate[0] and current_node[2][1] == self.start_coordinate[1]:
                planned_path.append(self.start_coordinate)
                break
        
        return planned_path[::-1]

if __name__ == "__main__":    

    # taking user input for start matrix and converting to np array
    print("Enter the start coordinate x y and theta:  ", end="")
    start_coordinate_input = input().split(" ")
    start_coordinate = (int(start_coordinate_input[0]), int(start_coordinate_input[1]))
    start_pos_orientation = float(start_coordinate_input[2])
    print("Your start coordinate",start_coordinate, start_pos_orientation)    

    #taking user input for start matrix and converting to np array
    print("Enter the goal coordinate x y:  ", end="")
    goal_coordinate_input = input().split(" ")
    goal_coordinate = (int(goal_coordinate_input[0]), int(goal_coordinate_input[1]))
    goal_pos_orientation = float(goal_coordinate_input[2])
    print("Your goal coordinate",goal_coordinate, goal_pos_orientation)

    #taking user input for robot radius
    print("Enter the robot radius:  ", end="")
    robot_radius = int(input())
    print("Your robot radius", robot_radius)

    #taking user input for map clearance
    print("Enter the map clearance:  ", end="")
    clearance = int(input())
    print("Your map clearance", clearance)

    #taking user input for step_size
    print("Enter the step size:  ", end="")
    step_size = int(input())
    print("Your step size", step_size)

    # numpy image canvas
    image = np.zeros((250, 400))
    hex_img = draw_hexagon((200, 100), 35, image)

    xpts = [36, 115, 80, 105]
    ypts = [185, 210, 180, 100]

    poly_img = draw_polygon(xpts, ypts, image)
    circle_img = draw_circle((300, 185), 40, image)

    # building map image by bitwise adding separate shape images
    map_image = cv2.bitwise_or(hex_img, poly_img)
    map_image = cv2.bitwise_or(map_image, circle_img)

    map_image = cv2.bitwise_or(map_image, draw_boundary(image))
    map_image = cv2.flip(map_image, 0)
    map_image = colorize_image(map_image, [47, 47, 211])

    circle_boundary = cv2.bitwise_xor(draw_circle((300, 185), 40, image), draw_circle((300, 185), 40 + clearance, image))
    hex_boundary = cv2.bitwise_xor(draw_hexagon((200, 100), 35, image), draw_hexagon((200, 100), 35+clearance, image))

    xpts_boundary = [26, 130, 90, 115]
    ypts_boundary = [185, 220, 180, 80]

    poly_boundary = cv2.bitwise_xor(draw_polygon(xpts, ypts, image), draw_polygon(xpts_boundary, ypts_boundary, image))

    # building boundary map image by bitwise adding separate shape images
    boundary_map_image = cv2.bitwise_or(circle_boundary, hex_boundary)
    boundary_map_image = cv2.bitwise_or(boundary_map_image, poly_boundary)
    boundary_map_image = cv2.flip(boundary_map_image, 0)

    final_map_img = overlay_boundary(map_image, boundary_map_image, [154, 154, 239])
    

    map_obj = MazeMap(final_map_img.copy(), robot_radius)

    planner = AStarPlanner(start_coordinate, goal_coordinate, map_obj, start_pos_orientation, goal_pos_orientation, step_size)
    planned_path, planned_path_img  = planner.search_for_path()

    if planned_path:
        for i in range(len(planned_path)):
            cv2.circle(planned_path_img, (planned_path[i][1], planned_path[i][0]), 2, (255, 0, 0), 2)
            if i > 0:
                cv2.line(planned_path_img, (planned_path[i-1][1], planned_path[i-1][0]), (planned_path[i][1], planned_path[i][0]), (255, 0, 0), 2)

        cv2.putText(planned_path_img, text='Steps: '+str(len(planned_path)), org=(300, 240), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 0, 0),thickness=1)
        cv2.imshow("Final planned path",planned_path_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    