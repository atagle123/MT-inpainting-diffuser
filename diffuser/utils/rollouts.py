import pickle

class TrajectoryBuffer:
    def __init__(self,first_observation,first_info):
        self.trajectories = []  # List to hold multiple trajectories
        self.start_trajectory(first_observation,first_info)

    def start_trajectory(self,first_observation,first_info):
        self.current_trajectory = {
            'states': [first_observation], # states has one dim more because of the next state... 
            'actions': [],
            'rewards': [],
            'dones': [],
            'total_reward':[],
            "infos":[first_info]# info has one dim more 
        }

    def add_transition(self, state, action, reward, done,total_reward,info):
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['dones'].append(done)
        self.current_trajectory['total_reward'].append(total_reward)
        self.current_trajectory['infos'].append(info)

    def end_trajectory(self):
        self.trajectories.append(self.current_trajectory)

    def save_trajectories(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.trajectories, f)

    def load_trajectories(self, filepath):
        with open(filepath, 'rb') as f:
            self.trajectories = pickle.load(f)