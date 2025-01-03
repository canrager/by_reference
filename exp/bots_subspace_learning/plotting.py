import matplotlib.pyplot as plt

def plot_training_history(train_history, val_history):
    """Plot training and validation losses over time."""
    plt.figure(figsize=(12, 6))
    
    # Plot total loss
    plt.subplot(1, 2, 1)
    plt.plot([x['total_loss'] for x in train_history], label='Train')
    plt.plot([x['total_loss'] for x in val_history], label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot component losses
    plt.subplot(1, 2, 2)
    plt.plot([x['align_loss'] for x in train_history], label='Train Align')
    plt.plot([x['separate_loss'] for x in train_history], label='Train Separate')
    plt.plot([x['rank_loss'] for x in train_history], label='Train Rank')
    plt.plot([x['align_loss'] for x in val_history], label='Val Align')
    plt.plot([x['separate_loss'] for x in val_history], label='Val Separate')
    plt.plot([x['rank_loss'] for x in val_history], label='Val Rank')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/training_history.png')
    plt.close()