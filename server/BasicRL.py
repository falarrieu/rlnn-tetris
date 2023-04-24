import numpy as np
import random
from BasicNN import NeuralNetwork

# Basic Reinforcement learning


# The state in the game of tetris needs to include the board and the type,position,rotation of current piece
# Actions are move right, left, rotate clockwise, rotate counterclockwise [,fast drop,hard drop]


valid_actions = [-1, 0, 1]


def epsilon_greedy(Qnet, state, valid_actions, epsilon):
    '''epsilon is between 0 and 1 and is the probability of returning a random action'''
    
    if np.random.uniform() < epsilon:
        # Random Move
        action = np.random.choice(valid_actions)
        
    else:
        # Greedy Move
        actions_randomly_ordered = random.sample(valid_actions, len(valid_actions))
        Qs = [Qnet.use(np.array([[state, a]])) for a in actions_randomly_ordered]
        ai = np.argmax(Qs)
        action = actions_randomly_ordered[ai]
        
    Q = Qnet.use(np.array([[state, action]]))
    
    return action, Q   # return the chosen action and Q(state, action)


def initial_state():
    # return np.random.randint(1, 11)
    pass

def next_state(s, a):
    # newstate = min(max(1, s + a), 10)
    # return newstate
    pass

def reinf(s, sn):  # sn is next state
    # return 0 if sn is 5 else -1  # if we have arrived in state 5, we have reached the goal, so r is 0.

    pass


# https://nbviewer.org/url/www.cs.colostate.edu/~cs445/notebooks/18.1%20Reinforcement%20Learning%20to%20Control%20a%20Marble.ipynb

# I think "make_samples" is misleading. I think it does technically do that, but I'd rather think of it
# as making moves in between training the neural network, since it's only try every once in a while
# In the case of the marble code, the environment is reset each time. It would probably make sense to do the same
# here, but have it complete a game of tetris each time. So it wouldn't be n steps per trial, just until it dies.
# Maybe we'd have a max number of steps once it gets better
def make_samples(Qnet, initial_state_f, next_state_f, reinforcement_f, valid_actions, n_samples, epsilon):

    X = np.zeros((n_samples, Qnet.n_inputs))
    R = np.zeros((n_samples, 1))
    Qn = np.zeros((n_samples, 1))

    s = initial_state_f()
    s = next_state_f(s, 0)        # New: Update state, sn from s and a
    a, _ = epsilon_greedy(Qnet, s, valid_actions, epsilon)

    # Collect data from n_samples steps
    for step in range(n_samples):
        
        sn = next_state_f(s, a)
        rn = reinforcement_f(s, sn)
        an, qn = epsilon_greedy(Qnet, sn, valid_actions, epsilon) # Forward pass for time t+1
        X[step, :] = np.hstack((s, a)) # accomodate multiple values in s
        R[step, 0] = rn
        Qn[step, 0] = qn
        # Advance one time step
        s, a = sn, an

    return (X, R, Qn)


def run(n_trials, n_steps_per_trial, n_epochs, learning_rate, n_hidden, gamma, epsilon_decay):
    epsilon = 1.0

    Qnet = NeuralNetwork(2, n_hidden, 1)

    # We need to set standardization parameters now so Qnet can be called to get first set of samples,
    # before it has been trained the first time.

    def setup_standardization(Qnet, Xmeans, Xstds, Tmeans, Tstds):
        Qnet.Xmeans = np.array(Xmeans)
        Qnet.Xstds = np.array(Xstds)
        Qnet.Tmeans = np.array(Tmeans)
        Qnet.Tstds = np.array(Tstds)

    # Inputs are position (1 to 10) and action (-1, 0, or 1)
    setup_standardization(Qnet, [5, 0], [2.5, 0.5], [0], [1])

    # fig = plt.figure(figsize=(10, 10))

    x_trace = np.zeros((n_trials * n_steps_per_trial, 2))
    r_trace = np.zeros((n_trials * n_steps_per_trial, 1))
    error_trace = []
    epsilon_trace = np.zeros((n_trials, 1))

    for trial in range(n_trials):
        
        X, R, Qn = make_samples(Qnet, initial_state, next_state, reinf, valid_actions, n_steps_per_trial, epsilon)
    
        T = R + gamma * Qn
        Qnet.train(X, T, n_epochs, learning_rate, method='adam')
        
        epsilon_trace[trial] = epsilon
        i = trial * n_steps_per_trial
        j = i + n_steps_per_trial
        x_trace[i:j, :] = X
        r_trace[i:j, :] = R
        error_trace += Qnet.error_trace
        
        epsilon *= epsilon_decay

        # Rest of this loop is for plots.
        if True and (trial + 1) % int(n_trials * 0.01 + 0.5) == 0:
            
            # fig.clf()
            # plot_status()
            # clear_output(wait=True)
            # display(fig)
        
    # clear_output(wait=True)
