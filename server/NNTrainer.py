import random
import math
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from Game import Game
from NeuralNetwork2 import DQN
from ReplayMemory import Transition, ReplayMemory

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    matplotlib.rcParams['figure.figsize'] = (20,10)
    
plt.ion() # Maybe remove??


class NNTrainer:
    
    def __init__(self,
                 batch_size,
                 gamma,
                 eps_start,
                 eps_end,
                 eps_decay,
                 tau,
                 learning_rate,
                 n_hiddens_per_layer,
                 num_episodes,
                 optimizer="adam",
                 points_state=True,
                 use_display=True
                 ):
        
        self.use_display = use_display
        
        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        if torch.cuda.is_available():
            self.num_episodes = 6000
        else:
            self.num_episodes = num_episodes
        
        self.game = Game()
        self.game.points_state = points_state
        
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TAU = tau
        self.LR = learning_rate

        self.n_actions = 5 # Number of possible actions
        # Get the number of state observations
        self.state = self.game.reset()
        # self.state, self.info = env.reset()
        self.n_observations = len(self.state)
        self.n_hiddens_per_layer = n_hiddens_per_layer

        self.policy_net = DQN(self.n_observations, self.n_hiddens_per_layer, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_hiddens_per_layer, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if optimizer == "adam":
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        else:
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=self.LR)
            
        self.memory = ReplayMemory(10000) # Maybe reduce this max capacity?


        self.steps_done = 0
        
        
        self.episode_durations = []
        self.loss_trace = []
        self.total_accuracy = 0
        self.accuracy_trace = []
        self.total_correct = 0
        self.correctness_trace = []


    def select_action(self, state, no_random=False):
        valid_actions = self.game.getValidActions()
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold or no_random:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                result = self.policy_net(state)
                result = torch.gather(result, 1, torch.LongTensor([valid_actions])) # Only allow actions that are valid
                return torch.tensor([[valid_actions[result.max(1)[1].view(1, 1).item()]]]) # Confusing, sorry
        else:
            return torch.tensor([[random.choice(valid_actions)]], device=self.device, dtype=torch.long)
            # return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)


    def plot_all(self, show_result=False):
        global is_ipython
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        loss_trace_t = torch.tensor(self.loss_trace, dtype=torch.float)
        
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
            
        # Durations plot
        dur_ax = plt.subplot(1,3,1)
        dur_ax.set_xlabel('Episode')
        dur_ax.set_ylabel('Duration')
        dur_ax.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            dur_ax.plot(means.numpy())
            
        # Loss plot
        loss_ax = plt.subplot(1,3,2)
        loss_ax.set_xlabel('Trial')
        loss_ax.set_ylabel('Loss')
        loss_ax.plot(loss_trace_t.numpy())
        # Take 100 episode averages and plot them too
        if len(loss_trace_t) >= 100:
            means = loss_trace_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            loss_ax.plot(means.numpy())
        
        # Tetris Image
        if not show_result:
            board_ax = plt.subplot(1,3,3)
            gray_map=plt.cm.get_cmap('gray')
            board = self.game.getBoard().board * 3 
            currentPoints = self.game.getPiece().getCurrentPoints()
            goalPoints = self.game.getGoalPiece().getCurrentPoints()
            for i in range(len(currentPoints)):
                board[goalPoints[i].y, goalPoints[i].x] = 1
                board[currentPoints[i].y, currentPoints[i].x] = 2
            board_ax.imshow(board, cmap=gray_map.reversed(), vmin=0, vmax=3)

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython and self.use_display:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
            
                
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        self.loss_trace.append(loss.item())
      
      
    def train(self):

        for i_episode in range(self.num_episodes):
            # Initialize the environment and get it's state
            state = self.game.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)

                # observation, reward, terminated, truncated, _ = env.step(action.item())
                observation, terminated = self.game.getNextFrame(action)
                reward, success = self.game.getReinforcements()
                reward = torch.tensor([reward], device=self.device)
                done = terminated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    if self.use_display: self.plot_all()
                    break

        print('Complete')
        finalFig = plt.subplot(1,3,1).figure
        self.plot_all(show_result=True)
        plt.ioff()
        finalFig.tight_layout(pad=0.1)
        finalFig.savefig("./figures/allPlots.png")
        
    def use(self):
        state = self.game.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        for t in count():
            action = self.select_action(state, no_random=True)
            
            observation, terminated = self.game.getNextFrame(action)
            done = terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Move to the next state
            state = next_state
            
            # self.plot_tetris()
            if done:
                self.accuracy_trace.append(self.game.calculate_piece_accuracy())
                self.total_correct += 1 if self.game.calculate_piece_accuracy() == 1 else 0
                self.correctness_trace.append(self.total_correct/ len(self.accuracy_trace))
                
                break
        # self.plot_tetris()
            
    def plot_tetris(self, show_result=False):
        global is_ipython
        plt.figure(1)
        
        # Tetris Image
        board_ax = plt.subplot(1,1,1)
        gray_map=plt.cm.get_cmap('gray')
        board = self.game.getBoard().board * 3 
        currentPoints = self.game.getPiece().getCurrentPoints()
        goalPoints = self.game.getGoalPiece().getCurrentPoints()
        for i in range(len(currentPoints)):
            board[goalPoints[i].y, goalPoints[i].x] = 1
            board[currentPoints[i].y, currentPoints[i].x] = 2
        board_ax.imshow(board, cmap=gray_map.reversed(), vmin=0, vmax=3)

        plt.pause(0.001)  # pause a bit so that plots are updated
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
    def save_model(self, file_path):
        torch.save(self.policy_net.state_dict(), file_path)

    def load_model(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))
        self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
