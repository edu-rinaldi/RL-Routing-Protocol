
import numpy as np
from scipy.spatial import distance
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt

class AIRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  #id event : (old_state, old_action)

        # Q-Table
        self.q_table = self.rnd_for_routing_ai.uniform(low=-2, high=0, size=[len(self.drone.path), self.simulator.n_drones])
        
        self.epsilon = 0.3
        self.alpha = 0.1
        self.gamma = 0.05

        self.old_state = None
        self.old_action = None
        self.old_reward = None

    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """
        # Packets that we delivered and still need a feedback
        # print(self.taken_actions)

        # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
        # Feedback from a delivered or expired packet
        print(drone, id_event, delay, outcome, self.get_reward(outcome, delay))


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
            
            # update using the old state and the selected action at that time
            # self.q_table[(state, ) + (action, )] += self.alpha * (reward + self.gamma * self.q_table[(new_state, ) + (new_action, )])
            # do something or train the model (?)

    def get_reward(self, outcome, delay):
        if outcome == -1:
            return -200
        return 20 * (1 - delay/self.simulator.event_duration)

    def get_state(self):
        return self.drone.path.index(self.drone.next_target())
        
    def get_action(self, state, available_drones):
        max_drone = None
        max_score = np.NINF
        for drone in available_drones:
            idx_drone = drone.identifier
            if self.q_table[(state, ) + (idx_drone, )] > max_score:
                max_drone = drone
                max_score = self.q_table[(state, ) + (idx_drone, )]
        # if max_drone.identifier == 5:
        #     print(state, available_drones, self.q_table[state])
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
        available_drones = [self.drone] + list(map(lambda x: x[1], opt_neighbors))
        action, best_drone = self.get_action(state, available_drones)
        # print(available_drones)
        # Store your current action --- you can add several stuff if needed to take a reward later
        self.taken_actions[pkd.event_ref.identifier] = (state, action)

        if self.rnd_for_routing_ai.rand() < self.epsilon:
            return self.rnd_for_routing_ai.choice(available_drones)

        return best_drone if best_drone.identifier != 0 else None

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        pass