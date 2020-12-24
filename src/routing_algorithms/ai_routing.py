
import numpy as np
from scipy.spatial import distance
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt

class AIRouting(BASE_routing):
    def __init__(self, drone, simulator, method='pos_perc', asm='egreedy'):
        """
            method:                         Position Based: 'pos_perc'   OR      Next Target Based: 'drone_path'
            asm (action selection method):  Epsilon Greedy: 'egreedy'    OR      Geo Epsilon Greedy: 'geo_egreedy'
        """
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  #id event : (old_state, old_action)

        # method for state representation
        self.method = method

        # action selection method
        self.asm = asm

        # Q-Table
        if method == 'drone_path':
            self.q_table = self.rnd_for_routing_ai.uniform(low=-2, high=0, size=[len(self.drone.path), self.simulator.n_drones])
        elif method == 'pos_perc':
            self.pos_precision = 20
            self.q_table = self.rnd_for_routing_ai.uniform(low=-2, high=0, size=[self.pos_precision, self.simulator.n_drones])
        self.epsilon = 0.08
        self.alpha = 0.2
        self.gamma = 0.4

        self.r_sum = 0
        self.timestep = 0
        self.r_avg = []

        self.old_state = None
        self.old_action = None
        self.old_reward = None

    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """
        # Packets that we delivered and still need a feedback
        # print(self.taken_actions)

        # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
        # Feedback from a delivered or expired packet
        # print(drone, id_event, delay, outcome, self.get_reward(outcome, delay))


        # remove the entry, the action has received the feedback
        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
        if id_event in self.taken_actions:
            state, action = self.taken_actions[id_event]
            del self.taken_actions[id_event]
            
            if self.old_state != None and self.old_action != None and self.old_reward != None:
                max_q = self.q_table[(state, ) + (action, )]
                self.q_table[(self.old_state, ) + (self.old_action, )] += self.alpha * (self.old_reward + (self.gamma * max_q) - self.q_table[(self.old_state, ) + (self.old_action, )] )    
            
            # state and action update
            self.old_state = state
            self.old_action = action
            # reward
            self.old_reward = self.get_reward(outcome, delay)
            self.r_sum += self.old_reward
            self.timestep += 1
            self.r_avg += [self.r_sum/self.timestep]
            

    def get_reward(self, outcome, delay):
        if outcome == -1:
            return -2
        return 2 * (1 - delay/self.simulator.event_duration)

    def get_state(self):
        if self.method == 'drone_path':
            return self.drone.path.index(self.drone.next_target())
        if self.method == 'pos_perc':
            perc = self.__get_curr_pos_percentage() * 100
            perc //= (100//self.pos_precision)
            return int(perc)
        
    def get_action(self, state, available_drones):
        max_drone = None
        max_score = np.NINF
        for drone in available_drones:
            idx_drone = drone.identifier
            if self.q_table[(state, ) + (idx_drone, )] > max_score:
                max_drone = drone
                max_score = self.q_table[(state, ) + (idx_drone, )]
        return max_drone.identifier, max_drone



    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """

        # Only if you need!
        # cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
        #                                                width_area=self.simulator.env_width,
        #                                                x_pos=self.drone.coords[0],  # e.g. 1500
        #                                                y_pos=self.drone.coords[1])[0]  # e.g. 500
        # print(cell_index)
        state = self.get_state()
        available_drones =  [self.drone] + list(map(lambda x: x[1], opt_neighbors))
        action, best_drone = self.get_action(state, available_drones)

        # Store your current action --- you can add several stuff if needed to take a reward later
        self.taken_actions[pkd.event_ref.identifier] = (state, action)
        
        if self.rnd_for_routing_ai.rand() < self.epsilon:
            if self.asm == 'geo_egreedy':
                # geo epsilon greedy action selection method

                # init best drone distance with distance between drone0 and the depot
                best_drone_distance = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords)
                best_drone = self.drone

                for hpk, drone_ist in opt_neighbors:
                    # calc expected position of the neighbor
                    expected_pos = self.__estimated_neighbor_drone_position(hpk)
                    # calc the expected distance from the depot 
                    expected_dist = util.euclidean_distance(expected_pos, self.simulator.depot.coords)
                    # get the drone with distance which is minimum
                    if expected_dist < best_drone_distance:
                        best_drone_distance = expected_dist
                        best_drone = drone_ist
                return best_drone
            else:
                # epsilon greedy
                return self.rnd_for_routing_ai.choice(available_drones)
        
        return best_drone

    def __get_path_size(self):
        total = 0
        # sum each distance and get the toal distance that the drone must do
        for i in range(len(self.drone.path)-1):
            p1 = self.drone.path[i]
            p2 = self.drone.path[i+1]
            total += util.euclidean_distance(p1, p2)
        return total

    def __get_path_size_point(self, point):
        total = 0
        f = False
        # sum each distance until the next point is the target one
        for i in range(len(self.drone.path)-1):
            p1 = self.drone.path[i]
            p2 = self.drone.path[i+1]
            if f: break
            if p2 == point: f = True

            total += util.euclidean_distance(p1, p2)
        return total

    def __get_curr_pos_percentage(self):
        total = self.__get_path_size()
        d = self.__get_path_size_point(self.drone.next_target()) - np.abs(util.euclidean_distance(self.drone.coords, self.drone.next_target()))
        return d/total

    def __estimated_neighbor_drone_position(self, hello_message):
        """ estimate the current position of the drone """

        # get known info about the neighbor drone
        hello_message_time = hello_message.time_step_creation
        known_position = hello_message.cur_pos
        known_speed = hello_message.speed
        known_next_target = hello_message.next_target

        # compute the time elapsed since the message sent and now
        # elapsed_time in seconds = elapsed_time in steps * step_duration_in_seconds
        elapsed_time = (self.simulator.cur_step - hello_message_time) * self.simulator.time_step_duration  # seconds

        # distance traveled by drone
        distance_traveled = elapsed_time * known_speed

        # direction vector
        a, b = np.asarray(known_position), np.asarray(known_next_target)
        v_ = (b - a) / np.linalg.norm(b - a)

        # compute the expect position
        c = a + (distance_traveled * v_)

        return tuple(c)

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        steps = np.arange(self.timestep)
        plt.plot(steps, self.r_avg)
        plt.ylabel("avg rewards")
        plt.show()


# ENT
class EGreedyNextTargetRouting(AIRouting):
    def __init__(self, drone, simulator):
        AIRouting.__init__(self, drone, simulator, 'drone_path', 'egreedy')

#GNT
class GeoNextTargetRouting(AIRouting):
    def __init__(self, drone, simulator):
        AIRouting.__init__(self, drone, simulator, 'drone_path', 'geo_egreedy')

#EPB
class EGreedyPositionBasedRouting(AIRouting):
    def __init__(self, drone, simulator):
        AIRouting.__init__(self, drone, simulator, 'pos_perc', 'egreedy')

#GPB
class GeoPositionBasedRouting(AIRouting):
    def __init__(self, drone, simulator):
        AIRouting.__init__(self, drone, simulator, 'pos_perc', 'geo_egreedy')