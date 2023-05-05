import numpy as np
import matplotlib.pyplot as plt
import time
from MovementNN import MovementNN

def main(): 
    
    n_trials = 1001
    n_steps_per_trial = 20
    n_epochs = 200
    learning_rate = 0.05
    n_hidden = [50, 25]
    gamma = 0.8
    final_epsilon = 0.0001  
    epsilon = 1.0
    trial_animations = [100, 500, 1000, 5000, 10000, 50000, 75000, 99997, 99998, 99999, 100000]
    
    
    movementNN = MovementNN(n_trials, n_steps_per_trial, n_epochs,
                            learning_rate, n_hidden, gamma, final_epsilon, epsilon, trial_animations)
    
    movementNN.Qnet.load()
    
    start = time.time()
    
    movementNN.train()
    
    end = time.time()
    print(f"Time: {end - start} s")
    
    movementNN.Qnet.save()
    
    # Error Trace
    plt.plot(movementNN.error_trace)
    plt.ylabel('TD Error')
    plt.xlabel('Epochs')
    plt.savefig("figures/tderror") 
    # plt.show()
    
    plt.clf()
    
    # Epsilon Trace
    plt.plot(movementNN.epsilon_trace)
    plt.ylabel('$\epsilon$')
    plt.xlabel('Trials')
    plt.ylim(0, 1)
    plt.savefig("figures/epsilontrace") 
    # plt.show()
    
    plt.clf()

    # R Trace
    plt.plot(movementNN.r_trace[: n_trials * n_steps_per_trial, 0])
    plt.ylabel('R')
    plt.xlabel('Steps')
    plt.savefig("figures/Rtrace")
    # plt.show()
    
    plt.clf()

    # R Smoothed
    plt.plot(np.convolve(movementNN.r_trace[:n_trials * n_steps_per_trial, 0], np.array([0.01] * 100), mode='valid'))
    plt.ylabel('R smoothed')
    # plt.ylim(-1.1, 0)
    plt.xlabel('Steps')
    plt.savefig("figures/Rsmoothed")
    # plt.show()
    
    plt.clf()
    
    # Attempts trace
    plt.plot(movementNN.attempts_trace)
    plt.ylabel('Attempts')
    plt.xlabel('Trials')
    plt.savefig("figures/attempts") 
    # plt.show()

    # movementNN.display_animations()
    np.save("animation_frames", movementNN.animation_frames)



if __name__ == "__main__":
    main()
