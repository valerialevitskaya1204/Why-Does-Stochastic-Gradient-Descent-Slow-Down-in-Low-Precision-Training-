import numpy as np
from algorithms.configs import PRECISIONS, BASE_LR, EPOCHS, FP4_LR, RESULTS_DIR
import matplotlib.pyplot as plt
import glob

def load_curves(precision, lr):
    files = glob.glob(f"{RESULTS_DIR}/sag_{precision}_lr{lr:.0e}_seed*.npz")
    losses, accs, qs = [], [], []
    for f in files:
        d = np.load(f)
        losses.append(d["train_loss"])
        accs.append(d["test_acc"])
        qs.append(d["shrinkage_q"])
    return np.mean(losses, axis=0), np.mean(accs, axis=0), np.mean(qs, axis=0)

def plot_results():
    epochs = np.arange(1, EPOCHS + 1)

    curves = {p: load_curves(p, BASE_LR) for p in PRECISIONS}
    fp4_big = load_curves("fp4", FP4_LR)

    plt.figure(figsize=(14, 6))

    #train loss
    plt.subplot(1, 4, 1)
    for p in PRECISIONS:
        plt.plot(epochs, curves[p][0], label=p.upper())
    plt.title("(A) Train Loss")
    plt.legend()
    plt.grid(True)

    #test accuracy
    plt.subplot(1, 4, 2)
    for p in PRECISIONS:
        plt.plot(epochs, curves[p][1], label=p.upper())
    plt.title("(B) Test Accuracy")
    plt.legend()
    plt.grid(True)

    #fp4 loss separate
    plt.subplot(1, 4, 3)
    plt.plot(epochs, curves["fp4"][0], label="FP4 (1e-4)")
    plt.plot(epochs, fp4_big[0], "--", label="FP4 (5e-4)")
    plt.title("(C) FP4 Train Loss")
    plt.legend()
    plt.grid(True)

    #q
    plt.subplot(1, 4, 4)
    for p in PRECISIONS:
        plt.plot(epochs, curves[p][2], label=p.upper())
    plt.title("(D) Mean Shrinkage q")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_everything(results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax1.set_title('A) Train Loss Comparison (lr=1e-4)')
    for name, data in results.items():
        if 'lr=1e-4' in name or name == 'FP32' or name == 'FP16':
            ax1.plot(data['train_losses'], label=name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('B) Test Accuracy Comparison (lr=1e-4)')
    for name, data in results.items():
        if 'lr=1e-4' in name or name == 'FP32' or name == 'FP16':
            ax2.plot(data['test_accuracies'], label=name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    ax3.set_title('C) FP4 Train Loss - Normal vs Increased Step Size')
    if 'FP4 (lr=1e-4)' in results:
        ax3.plot(results['FP4 (lr=1e-4)']['train_losses'], label='FP4 (lr=1e-4)', linestyle='--')
    if 'FP4 (lr=5e-4)' in results:
        ax3.plot(results['FP4 (lr=5e-4)']['train_losses'], label='FP4 (lr=5e-4)', linestyle='-')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train Loss')
    ax3.legend()
    ax3.grid(True)
    
    ax4.set_title('D) FP4 Test Accuracy - Normal vs Increased Step Size')
    if 'FP4 (lr=1e-4)' in results:
        ax4.plot(results['FP4 (lr=1e-4)']['test_accuracies'], label='FP4 (lr=1e-4)', linestyle='--')
    if 'FP4 (lr=5e-4)' in results:
        ax4.plot(results['FP4 (lr=5e-4)']['test_accuracies'], label='FP4 (lr=5e-4)', linestyle='-')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/resnet50_cifar10_precision_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(15, 12))
    
    ax5.set_title('E) Train Loss (Epochs 60-100)')
    for name, data in results.items():
        if 'lr=1e-4' in name or name == 'FP32' or name == 'FP16':
            epochs = range(len(data['train_losses']))
            ax5.plot(epochs[60:100], data['train_losses'][60:100], label=name)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Train Loss')
    ax5.set_xlim(60, 99)
    ax5.legend()
    ax5.grid(True)
    
    ax6.set_title('F) Test Accuracy (Epochs 60-100)')
    for name, data in results.items():
        if 'lr=1e-4' in name or name == 'FP32' or name == 'FP16':
            epochs = range(len(data['test_accuracies']))
            ax6.plot(epochs[60:100], data['test_accuracies'][60:100], label=name)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Test Accuracy (%)')
    ax6.set_xlim(60, 99)
    ax6.legend()
    ax6.grid(True)
    
    ax7.set_title('G) FP4 Train Loss (Epochs 60-100)')
    if 'FP4 (lr=1e-4)' in results:
        epochs = range(len(results['FP4 (lr=1e-4)']['train_losses']))
        ax7.plot(epochs[60:100], results['FP4 (lr=1e-4)']['train_losses'][60:100], 
                label='FP4 (lr=1e-4)', linestyle='--')
    if 'FP4 (lr=5e-4)' in results:
        epochs = range(len(results['FP4 (lr=5e-4)']['train_losses']))
        ax7.plot(epochs[60:100], results['FP4 (lr=5e-4)']['train_losses'][60:100], 
                label='FP4 (lr=5e-4)', linestyle='-')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Train Loss')
    ax7.set_xlim(60, 99)
    ax7.legend()
    ax7.grid(True)
    
    ax8.set_title('H) FP4 Test Accuracy (Epochs 60-100)')
    if 'FP4 (lr=1e-4)' in results:
        epochs = range(len(results['FP4 (lr=1e-4)']['test_accuracies']))
        ax8.plot(epochs[60:100], results['FP4 (lr=1e-4)']['test_accuracies'][60:100], 
                label='FP4 (lr=1e-4)', linestyle='--')
    if 'FP4 (lr=5e-4)' in results:
        epochs = range(len(results['FP4 (lr=5e-4)']['test_accuracies']))
        ax8.plot(epochs[60:100], results['FP4 (lr=5e-4)']['test_accuracies'][60:100], 
                label='FP4 (lr=5e-4)', linestyle='-')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Test Accuracy (%)')
    ax8.set_xlim(60, 99)
    ax8.legend()
    ax8.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}resnet50_cifar10_precision_comparison_epochs_60_100.png', dpi=300, bbox_inches='tight')
    plt.show()
 