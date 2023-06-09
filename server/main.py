import numpy as np
import matplotlib.pyplot as plt
import time
from MovementNN import MovementNN
from NNTrainer import NNTrainer
import os

def main(): 
    
    n_trials = 1001
    n_steps_per_trial = 20
    n_epochs = 200
    learning_rate = 0.05
    n_hidden = [512, 256, 128]
    gamma = 0.4 #0.8
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


def main2():
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    n_hiddens_per_layer=[512, 256, 128]
    num_episodes = 100000

    trainer = NNTrainer(
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay=EPS_DECAY,
        tau=TAU,
        learning_rate=LR,
        n_hiddens_per_layer=n_hiddens_per_layer,
        num_episodes=num_episodes,
        use_display=False,
    )

    file_path = "saved_model.pth"

    if os.path.exists(file_path):
        trainer.load_model(file_path)

    trainer.train()

    trainer.save_model(file_path)

if __name__ == "__main__":
    main2()
