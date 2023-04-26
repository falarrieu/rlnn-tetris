from NeuralNetwork import NeuralNetwork
import numpy as np
import random
from Game import Game

class MovementNN:
    
    def __init__(self, n_trials, n_steps_per_trial, n_epochs, learning_rate,
                 n_hidden, gamma, final_epsilon, epsilon, trial_animations):
        
        self.game = Game()
        self.valid_actions = []
        self.trial_animations = trial_animations
        
        '''Initialization Parameters'''        
        self.n_trials = n_trials
        self.n_steps_per_trial = n_steps_per_trial
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.gamma = gamma
        self.final_epsilon = final_epsilon  
        self.epsilon = epsilon
        self.epsilon_decay = np.exp(np.log(self.final_epsilon) / self.n_trials)
        
        '''Instantiate Neural Network'''
        self.Qnet = NeuralNetwork(18, self.n_hidden, 1)
        
        self.Xmeans = [np.mean(np.arange(0,2**19-1))] * 10
        self.Xmeans.extend([np.mean(np.arange(0,10)), np.mean(np.arange(0,20))] * 4)
        self.Xstds = [np.std(np.arange(0,2**19-1))] * 10
        self.Xstds.extend([np.std(np.arange(0,10)), np.std(np.arange(0,20))] * 4)
        self.setup_standardization(self.Qnet, self.Xmeans, self.Xstds, [5], [1])
        
        self.x_trace = np.zeros((self.n_trials * self.n_steps_per_trial, 18))
        self.r_trace = np.zeros((self.n_trials * self.n_steps_per_trial, 1))
        self.error_trace = []
        self.epsilon_trace = np.zeros((self.n_trials, 1))

        '''Visualizing'''
        self.animation_frames = []

        pass
    
    def setup_standardization(self, Qnet, Xmeans, Xstds, Tmeans, Tstds):
        '''We need to set standardization parameters now so Qnet can be called to get first set of samples,
        before it has been trained the first time.'''
        Qnet.Xmeans = np.array(Xmeans)
        Qnet.Xstds = np.array(Xstds)
        Qnet.Tmeans = np.array(Tmeans)
        Qnet.Tstds = np.array(Tstds)
    
    def epsilon_greedy(self, Qnet, state, valid_actions, epsilon):
        '''epsilon is between 0 and 1 and is the probability of returning a random action'''
        valid_actions = self.game.getValidActions()
        if np.random.uniform() < epsilon:
            # Random Move
            action = np.random.choice(valid_actions)
            
        else:
            # Greedy Move
            actions_randomly_ordered = random.sample(valid_actions, len(valid_actions))
            Qs = [Qnet.use(np.array(state)) for a in actions_randomly_ordered]
            ai = np.argmax(Qs)
            action = actions_randomly_ordered[ai]
            
        Q = Qnet.use(np.array(state))
        
        return action, Q   # return the chosen action and Q(state, action)
    
    def use(self, X):
        self.Qnet.use(X)
    
    def initial_state(self):   
        return self.game.getBoard(), self.game.getPiece().copy() 

    def next_state(self, s, a):
        return self.game.getNextFrame(s, a)

    def reinf(self, s, sn):  # sn is next state
        return self.game.getReinforcements(s, sn)
    
    def make_samples(self, Qnet, initial_state_f, next_state_f, reinforcement_f,
                 valid_actions, n_samples, epsilon, trial_num):

        X = np.zeros((n_samples, Qnet.n_inputs))
        R = np.zeros((n_samples, 1))
        Qn = np.zeros((n_samples, 1))

        s = initial_state_f()
        expanded_state = self.expand_state(s)
        a, _ = self.epsilon_greedy(Qnet, expanded_state, valid_actions, epsilon)

        frames = []
        # Collect data from n_samples steps
        for step in range(n_samples):
            
            next_state = next_state_f(s, a)        # Update state, sn, from s and a
            sn = next_state[:2]
            will_lock = next_state[2]
            if will_lock:
                self.game.lockPiece()
                break
            rn = reinforcement_f(s, sn)    # Calculate resulting reinforcement
            an, qn = self.epsilon_greedy(Qnet, self.expand_state(sn), valid_actions, epsilon)  # choose next action
    #         print(an)
            X[step, :] = self.expand_state(s)
            R[step, 0] = rn
            Qn[step, 0] = qn
            s, a = sn, an  # Advance one time step
            
            frame = (step, self.game.getPiece().copy(), self.game.goalPiece.copy(), trial_num)
            frames.append(frame)

        if trial_num in self.trial_animations:
            self.animation_frames.append([trial_num, *frames])
            # self.game.getBoard().createAnimations(frames, trial_num)

        return (X, R, Qn)
    
    def expand_state(self, s):
        '''Every square on the board has a 0 if empty or 1 if filled. So we take each column, make a binary string from
        top to bottom, and convert to decimal to create the inputs to the NN. This is great because no information
        is lost about the column, and you can determine whether a column is higher than another just by comparing
        the value. Hopefully This will help to preserve patterns'''
        board, piece = s
        decimal_columns = [int(''.join([str(int(item)) for item in row]), 2) for row in board.board.T]
        flat_piece_points = np.reshape([[point.x,point.y] for point in piece.getCurrentPoints()], 8)
        return np.hstack((decimal_columns, flat_piece_points))
    
    def train(self):
        
        for trial in range(self.n_trials):
            print("Trial: ", trial)
            
            X, R, Qn = self.make_samples(self.Qnet, self.initial_state, self.next_state, self.reinf, self.valid_actions, self.n_steps_per_trial, self.epsilon, trial)
        
            T = R + self.gamma * Qn
            self.Qnet.train(X, T, self.n_epochs, self.learning_rate, method='adam', verbose=False)
            
            self.epsilon_trace[trial] = self.epsilon
            self.epsilon *= self.epsilon_decay
            i = trial * self.n_steps_per_trial
            j = i + self.n_steps_per_trial
            self.x_trace[i:j, :] = X
            self.r_trace[i:j, :] = R
            self.error_trace += self.Qnet.error_trace
            
            self.game.nextPiece()

    def display_animations(self):
        for animation in self.animation_frames:
            trial_num = animation[0]
            frames = animation[1:]
            self.game.getBoard().createAnimations(frames, trial_num)
        