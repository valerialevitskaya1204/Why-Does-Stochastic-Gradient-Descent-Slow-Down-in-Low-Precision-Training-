from training.train import run_experiments, train_baseline
from training.plots.plot_smth import plot_results, plot_everything


if __name__ == "__main__":
    run_experiments() #for saga
    plot_results() #for saga
    
    results = train_baseline() #baseline
    plot_everything(results) #baseline
    