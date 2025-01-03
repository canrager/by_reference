import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
from typing import Tuple, Dict
from subspace_training import EntitySubspaceLearner


def analyze_subspace(
    Pe: torch.Tensor,
    entity_vectors: torch.Tensor,
    entity_labels: torch.Tensor,
    n_entities: int = 3
) -> Tuple[float, float, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analyze the learned subspace properties."""

    # Normalize and project vectors
    entity_vectors = normalize(entity_vectors, dim=-1)
    proj_entities = entity_vectors @ Pe.T
    proj_entities = normalize(proj_entities, dim=-1)
    
    # Calculate intra-cluster distances
    intra_dists = torch.zeros(n_entities)
    for i in range(n_entities):
        cluster = proj_entities[entity_labels == i]
        # intra_dists[i] = torch.pdist(cluster).mean() # euclidean distance
        unique_pairs_mask = torch.triu(torch.ones(cluster.shape[0], cluster.shape[0]), diagonal=1).bool()
        intra_dists[i] = torch.mm(cluster, cluster.T)[unique_pairs_mask].mean() # cosine distance
    intra_cluster_dist = intra_dists.mean()
    print(intra_cluster_dist)
    
    # Calculate inter-cluster distances
    inter_dists = []
    for i in range(n_entities):
        cluster_i = proj_entities[entity_labels == i]
        for j in range(i + 1, n_entities):
            cluster_j = proj_entities[entity_labels == j]
            # dist = torch.cdist(
            #     cluster_i,
            #     cluster_j,
            # ).mean()
            dist = torch.mm(cluster_i, cluster_j.T).mean()
            inter_dists.append(dist)
    inter_cluster_dist = torch.tensor(inter_dists).mean()
    
    # Calculate effective rank
    U, S, V = torch.svd(Pe)
    total_variance = torch.sum(S)
    cumulative_variance = torch.cumsum(S, dim=0)
    effective_rank = torch.sum(cumulative_variance < 0.90 * total_variance).item()

    U, S, V = U.detach().cpu(), S.detach().cpu(), V.detach().cpu()
    
    return intra_cluster_dist, inter_cluster_dist, effective_rank, U, S, V

def analyze_cluster_metrics(proj_orig: np.ndarray, proj_pe: np.ndarray) -> Dict[str, float]:
    """Compute cluster separation metrics for both original and projected spaces."""
    metrics = {}
    
    # Calculate metrics for both projections
    for name, proj in [('original', proj_orig), ('pe', proj_pe)]:
        # Intra-cluster distances
        intra_dists = []
        for i in range(proj.shape[1]):
            cluster = proj[:, i, :]
            dists = torch.pdist(torch.tensor(cluster))
            intra_dists.append(dists.mean().item())
        metrics[f'{name}_intra_dist'] = np.mean(intra_dists)
        
        # Inter-cluster distances
        inter_dists = []
        for i in range(proj.shape[1]):
            for j in range(i + 1, proj.shape[1]):
                dist = np.linalg.norm(
                    proj[:, i, :].mean(axis=0) - proj[:, j, :].mean(axis=0)
                )
                inter_dists.append(dist)
        metrics[f'{name}_inter_dist'] = np.mean(inter_dists)
        
        # Separability ratio
        metrics[f'{name}_separability'] = metrics[f'{name}_inter_dist'] / metrics[f'{name}_intra_dist']
    
    return metrics

def compare_pca_spaces(
    P_e: torch.Tensor, 
    entity_vectors: torch.Tensor, 
    entity_labels: torch.Tensor,
    save_path: str = 'images/pca_comparison.png',
    num_entities: int = 3
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compare PCA results in original and P_e projected spaces."""
    with torch.no_grad():
        # Normalize input vectors
        entities_R = normalize(entity_vectors, dim=-1)
        
        # Project vectors through P_e
        entities_P = entity_vectors @ P_e.T
        
        # Perform PCA in both spaces
        pca_orig = PCA(n_components=2)
        pca_pe = PCA(n_components=2)
        
        # Transform data
        proj_orig = pca_orig.fit_transform(entities_R.cpu().numpy())
        proj_pe = pca_pe.fit_transform(entities_P.cpu().numpy())
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Colors and labels for entities
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = ['Entity 1', 'Entity 2', 'Entity 3']
        
        # Plot original space
        for i in range(num_entities):
            entity_points = proj_orig[entity_labels == i]
            ax1.scatter(
                entity_points[:, 0],
                entity_points[:, 1],
                c=colors[i],
                label=labels[i],
                alpha=0.6
            )
        ax1.set_title(f'Original Space\nExplained Variance: {pca_orig.explained_variance_ratio_.sum():.2%}')
        ax1.set_xlabel('First Principal Component')
        ax1.set_ylabel('Second Principal Component')
        ax1.legend()
        
        # Plot P_e space
        for i in range(num_entities):
            entity_points = proj_pe[entity_labels == i]
            ax2.scatter(
                entity_points[:, 0],
                entity_points[:, 1],
                c=colors[i],
                label=labels[i],
                alpha=0.6
            )
        ax2.set_title(f'P_e Projected Space\nExplained Variance: {pca_pe.explained_variance_ratio_.sum():.2%}')
        ax2.set_xlabel('First Principal Component')
        ax2.set_ylabel('Second Principal Component')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return {
            'orig_explained_var': pca_orig.explained_variance_ratio_,
            'pe_explained_var': pca_pe.explained_variance_ratio_
        }, pca_pe.components_

def visualize_singular_values(S: torch.Tensor, save_path: str = 'images/singular_values.png'):
    """Visualize the singular values of the projection matrix."""
    s_numpy = S.cpu().numpy()
    s_normalized = s_numpy / s_numpy.sum()
    
    plt.figure(figsize=(8, 6))
    plt.plot(s_normalized, marker='o')
    plt.title("Singular Values of Projection Matrix")
    plt.xlabel("Index")
    plt.ylabel("Normalized Singular Value")
    plt.savefig(save_path)
    plt.close()

def visualize_entity_projections(
    entity_vectors: torch.Tensor,
    entity_labels: torch.Tensor,
    V: torch.Tensor,
    save_dir: str = 'images',
    num_entities: int = 3,
):
    """Visualize entity projections using singular vectors."""
    # Colors for different entities
    colors = plt.cm.rainbow(np.linspace(0, 1, num_entities))
    
    # Project onto right singular vectors (V)
    projected_v = entity_vectors @ V[:, :2]
    
    plt.figure(figsize=(10, 8))
    for i in range(num_entities):
        entity_points = projected_v[entity_labels == i]
        plt.scatter(
            entity_points[:, 0],
            entity_points[:, 1],
            c=[colors[i]],
            label=f'Entity {i+1}',
            alpha=0.6
        )
    
    plt.title('Entity Vectors Projected onto First Two Right Singular Vectors (V)')
    plt.xlabel('First Right Singular Vector')
    plt.ylabel('Second Right Singular Vector')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/entity_projections_V.png')
    plt.close()

def plot_training_history(train_history: list, val_history: list, learner: EntitySubspaceLearner):
    """Plot training and validation loss history."""
    plt.figure(figsize=(12, 8))
    
    # Plot each loss component
    loss_titles = {
    'align_loss': f'align loss (before coeff {learner.align_coeff})',
    'separate_loss': f'separate loss(before coeff {learner.separate_coeff})',
    'rank_loss': f'rank loss (before coeff {learner.rank_coeff})',
    'total_loss': 'total loss'
    }
    
    loss_types = list(loss_titles.keys())
    for loss_type in loss_types:
        plt.subplot(2, 2, loss_types.index(loss_type) + 1)
        
        # Plot training loss
        train_losses = [epoch[loss_type] for epoch in train_history]
        plt.plot(train_losses, label='Train')
        
        # Plot validation loss
        val_losses = [epoch[loss_type] for epoch in val_history]
        plt.plot(val_losses, label='Validation')
        
        plt.title(f'{loss_titles[loss_type].title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig('images/training_history.png')
    plt.close()

def visualize_pca_many_components(t: torch.Tensor, i: torch.Tensor, n_components: int = 5, save_prefix: str = "pca_plot"):
    """
    Perform PCA on a batch of data and create scatterplots of paired principal components.
    
    Args:
        t: Input tensor of shape (batch_size, dim)
        i: Class label tensor of shape (batch_size,) with 1-indexed integers
        n_components: Number of principal components to compute
        save_prefix: Prefix for saved plot filenames
    """
    # Convert tensors to numpy arrays
    X = t.detach().cpu().numpy()
    labels = i.detach().cpu().numpy() - 1  # Convert to 0-indexed
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # Get unique classes for color mapping
    unique_classes = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    
    # Create scatterplots for pairs of components
    for idx in range(0, n_components-1, 2):
        pc1, pc2 = idx, idx + 1
        
        plt.figure(figsize=(10, 8))
        for class_idx, color in zip(unique_classes, colors):
            mask = labels == class_idx
            plt.scatter(X_pca[mask, pc1], X_pca[mask, pc2], 
                       c=[color], label=f'Class {class_idx + 1}',
                       alpha=0.7)
        
        # Add labels and title
        plt.xlabel(f'Principal Component {pc1 + 1}')
        plt.ylabel(f'Principal Component {pc2 + 1}')
        plt.title(f'PCA Components {pc1 + 1} vs {pc2 + 1}')
        
        # Add variance explained information
        var_explained = pca.explained_variance_ratio_
        plt.text(0.02, 0.98, 
                f'Variance explained:\nPC{pc1 + 1}: {var_explained[pc1]:.3f}\nPC{pc2 + 1}: {var_explained[pc2]:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(f'images/pca_R/{save_prefix}_components_{pc1+1}_{pc2+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Print total variance explained
    total_var_explained = np.sum(pca.explained_variance_ratio_)
    print(f'Total variance explained by {n_components} components: {total_var_explained:.3f}')
    
    return pca