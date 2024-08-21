import pickle

class TrajectoryBuffer:
    def __init__(self):
        self.trajectories = []  # List to hold multiple trajectories
        self.start_trajectory()

    def start_trajectory(self):
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
            'total_reward':[]
        }

    def add_transition(self, state, action, reward, next_state, done,total_reward):
        self.current_trajectory['states'].append(state)
        self.current_trajectory['actions'].append(action)
        self.current_trajectory['rewards'].append(reward)
        self.current_trajectory['next_states'].append(next_state)
        self.current_trajectory['dones'].append(done)
        self.current_trajectory['total_reward'].append(total_reward)

    def end_trajectory(self):
        self.trajectories.append(self.current_trajectory)

    def save_trajectories(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.trajectories, f)

    def load_trajectories(self, filepath):
        with open(filepath, 'rb') as f:
            self.trajectories = pickle.load(f)