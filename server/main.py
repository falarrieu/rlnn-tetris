import numpy as np
import matplotlib.pyplot as plt
from MovementNN import MovementNN

def main(): 
    
    n_trials = 11
    n_steps_per_trial = 20
    n_epochs = 200
    learning_rate = 0.05
    n_hidden = [50, 25]
    gamma = 0.8
    final_epsilon = 0.0001  
    epsilon = 1.0
    trial_animations = [1, 10, 100, 500, 1000, 5000, 10000, 50000, 75000, 99997, 99998, 99999, 100000]
    
    
    movementNN = MovementNN(n_trials, n_steps_per_trial, n_epochs,
                            learning_rate, n_hidden, gamma, final_epsilon, epsilon, trial_animations)
    
    movementNN.train()
    
    # Error Trace
    plt.plot(np.arange(0, n_trials * n_epochs), movementNN.error_trace)
    plt.show()
    
    # Epsilon Trace
    plt.plot(np.arange(0, n_trials), movementNN.epsilon_trace)
    plt.show()
    
    # X Trace
    plt.plot(np.arange(0, n_trials * n_epochs), movementNN.x_trace)
    plt.show()
    
    # R Trace
    plt.plot(np.arange(0, n_trials * n_epochs), movementNN.r_trace)
    plt.show()

    movementNN.display_animations()



if __name__ == "__main__":
    main()