import json
import matplotlib.pyplot as plt

def load_data(file="genetic_results.json"):
    with open(file, "r") as f:
        return json.load(f)

def plot_fitness(history):
    generations = [gen["generation"] for gen in history]
    best_fitness = [gen["best"]["fitness"] for gen in history]
    avg_fitness = [
        sum(ind["fitness"] for ind in gen["individuals"]) / len(gen["individuals"])
        for gen in history
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(generations, best_fitness, label="Best Fitness", marker="o")
    plt.plot(generations, avg_fitness, label="Average Fitness", linestyle="--")
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitness_over_generations.png")

def plot_weight_evolution(history):
    plt.figure(figsize=(10, 5)) 
    num_weights = len(history[0]["best"]["weights"])
    generations = [gen["generation"] for gen in history]

    weight_labels = ["Lines", "Holes", "Depth", "Density"]

    # For each weight index, track evolution of its value
    for i in range(num_weights):
        weight_values = [gen["best"]["weights"][i] for gen in history]

        plt.plot(generations, weight_values, label=f"{weight_labels[i]} Weight")

    plt.title("Best Weights over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Weight Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("weights_over_generations.png")

if __name__ == "__main__":
    history = load_data()
    plot_fitness(history)
    plot_weight_evolution(history)
