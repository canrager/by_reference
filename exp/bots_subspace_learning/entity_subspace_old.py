import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
import seaborn as sns
from tqdm import tqdm
from typing import Tuple, Dict

from plotting import plot_training_history
from pca_utils import compare_pca_spaces, analyze_cluster_metrics

class EntitySubspaceLearner(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        subspace_dim: int = 32,
        margin: float = 0.5,
        rank_weight: float = 0.1,  # Weight for the rank regularization term
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        force_orthogonal: bool = True,
    ):
        super().__init__()
        self.device = device
        self.subspace_dim = subspace_dim
        self.margin = margin
        self.rank_weight = rank_weight
        self.force_orthogonal = force_orthogonal
        
        # Initialize projection matrix as a proper nn.Parameter
        P_e = torch.empty(subspace_dim, input_dim, device=device)
        nn.init.orthogonal_(P_e)
        self.P_e = nn.Parameter(P_e)
        
        # Move to device
        self.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def compute_rank_loss(self) -> torch.Tensor:
        """Compute nuclear norm (trace norm) of the projection matrix to encourage low rank."""
        # Compute singular values of P_e
        _, S, _ = torch.svd(self.P_e)
        # Nuclear norm is the sum of singular values
        nuclear_norm = torch.sum(S)
        return nuclear_norm
    
    def entity_alignment_loss(self, entity_vectors: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss for entities with same index.
        
        Args:
            entity_vectors: Shape (batch_size, num_entities, hidden_dim)
        """
        batch_size, num_entities, _ = entity_vectors.shape
        
        # Normalize input vectors
        entity_vectors = normalize(entity_vectors, dim=-1)
        
        # Project to subspace
        # Reshape to (batch_size * num_entities, hidden_dim)
        flat_vectors = entity_vectors.reshape(-1, entity_vectors.shape[-1])
        proj_entities = self.P_e @ flat_vectors.t()  # Shape: (subspace_dim, batch_size * num_entities)
        proj_entities = proj_entities.t()  # Shape: (batch_size * num_entities, subspace_dim)
        
        # Reshape back to (batch_size, num_entities, subspace_dim)
        proj_entities = proj_entities.reshape(batch_size, num_entities, self.subspace_dim)
        
        # Normalize projected vectors
        proj_entities = normalize(proj_entities, dim=-1)
        
        loss = 0.0
        # For each entity index, compute mean cosine similarity within cluster
        for entity_idx in range(num_entities):
            entity_cluster = proj_entities[:, entity_idx, :]  # Shape: (batch_size, subspace_dim)
            # Compute pairwise cosine similarities within cluster
            sim_matrix = torch.mm(entity_cluster, entity_cluster.t())  # Shape: (batch_size, batch_size)
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
            sim_matrix = sim_matrix[mask].reshape(batch_size, batch_size - 1)
            # Maximize similarity (minimize negative similarity)
            loss -= sim_matrix.mean() # Optimal loss is -1.0
            
        normalized_loss = loss / num_entities
        normalized_loss += 1.0  # Normalize to [0, 2]
        
        return normalized_loss / num_entities
    
    def entity_separation_loss(self, entity_vectors: torch.Tensor) -> torch.Tensor:
        """Compute separation loss between different entity indices.
        
        Args:
            entity_vectors: Shape (batch_size, num_entities, hidden_dim)
        """
        batch_size, num_entities, _ = entity_vectors.shape
        
        # Normalize and project vectors as before
        entity_vectors = normalize(entity_vectors, dim=-1)
        flat_vectors = entity_vectors.reshape(-1, entity_vectors.shape[-1])
        proj_entities = self.P_e @ flat_vectors.t()
        proj_entities = proj_entities.t()
        proj_entities = proj_entities.reshape(batch_size, num_entities, self.subspace_dim)
        proj_entities = normalize(proj_entities, dim=-1)
        
        loss = 0.0
        # Compare each pair of different entity indices
        for i in range(num_entities):
            for j in range(i + 1, num_entities):
                # Compute cosine similarity between clusters
                sim = torch.mm(proj_entities[:, i, :], proj_entities[:, j, :].t())
                # Apply margin-based loss
                loss += torch.relu(sim - self.margin).mean()
                
        return loss
    
    def train_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        # Unpack batch
        entity_vectors = batch[0].to(self.device)
        self.optimizer.zero_grad()
        
        # Compute losses
        align_loss = self.entity_alignment_loss(entity_vectors)
        separate_loss = self.entity_separation_loss(entity_vectors)
        rank_loss = self.compute_rank_loss()
        
        # Total loss with rank regularization
        total_loss = align_loss + separate_loss + self.rank_weight * rank_loss
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Project P_e back to the Stiefel manifold (orthogonality constraint)
        if self.force_orthogonal:
            with torch.no_grad():
                U, _, V = torch.svd(self.P_e)
                self.P_e.data = torch.mm(U, V.t())
        
        return {
            'align_loss': align_loss.item(),
            'separate_loss': separate_loss.item(),
            'rank_loss': rank_loss.item(),
            'total_loss': total_loss.item()
        }

    def analyze_subspace(self, entity_vectors: torch.Tensor) -> Tuple[float, float, int]:
        """Analyze the learned subspace."""
        with torch.no_grad():
            # Normalize and project vectors
            entity_vectors = normalize(entity_vectors, dim=-1)
            flat_vectors = entity_vectors.reshape(-1, entity_vectors.shape[-1])
            proj_entities = self.P_e @ flat_vectors.t()
            proj_entities = proj_entities.t()
            proj_entities = proj_entities.reshape(
                entity_vectors.shape[0], 
                entity_vectors.shape[1], 
                self.subspace_dim
            )
            proj_entities = normalize(proj_entities, dim=-1)
            
            # Calculate intra-cluster distances
            intra_dists = []
            for i in range(entity_vectors.shape[1]):
                cluster = proj_entities[:, i, :]
                dists = torch.pdist(cluster)
                intra_dists.append(dists.mean().item())
            intra_cluster_dist = np.mean(intra_dists)
            
            # Calculate inter-cluster distances
            inter_dists = []
            for i in range(entity_vectors.shape[1]):
                for j in range(i + 1, entity_vectors.shape[1]):
                    dist = torch.cdist(
                        proj_entities[:, i, :],
                        proj_entities[:, j, :]
                    ).mean()
                    inter_dists.append(dist.item())
            inter_cluster_dist = np.mean(inter_dists)
            
            # Calculate effective rank and get singular vectors
            U, S, V = torch.svd(self.P_e)
            total_variance = torch.sum(S)
            cumulative_variance = torch.cumsum(S, dim=0)
            effective_rank = torch.sum(cumulative_variance < 0.90 * total_variance).item()

            # Save right singular vectors for visualization
            torch.save(V, "artifacts/V_singular_vectors.pt")

            # Plot singular values
            s_numpy = S.cpu().numpy()
            s_normalized = s_numpy / s_numpy.sum()
            plt.figure(figsize=(8, 6))
            plt.plot(s_normalized, marker='o')
            plt.title("Singular Values of Projection Matrix")
            plt.xlabel("Index")
            plt.ylabel("Singular Value")
            plt.savefig("images/singular_values.png")
            plt.close()

            # Create scatterplot of entity vectors projected onto first two right singular vectors (V)
            plt.figure(figsize=(10, 8))
            
            # Project original vectors onto right singular vectors
            V = V.cpu()
            original_vectors = entity_vectors.cpu()
            batch_size, num_entities, hidden_dim = original_vectors.shape
            
            # Reshape to 2D for projection
            original_flat = original_vectors.reshape(-1, hidden_dim)
            # Project onto first two right singular vectors
            projected_v = original_flat @ V[:, :2]
            
            # Plot each entity class with different colors
            colors = plt.cm.rainbow(np.linspace(0, 1, num_entities))
            for i in range(num_entities):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                entity_points = projected_v[start_idx:end_idx]
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
            plt.savefig('images/entity_projections_V.png')
            plt.close()

            # Create scatterplot of P_e-projected vectors along first two left singular vectors (U)
            plt.figure(figsize=(10, 8))
            
            # Get P_e-projected vectors and reshape
            proj_flat = proj_entities.reshape(-1, self.subspace_dim).cpu()
            # Project onto first two left singular vectors
            U = U.cpu()
            projected_u = proj_flat @ U[:, :2]
            
            # Plot each entity class with different colors
            for i in range(num_entities):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                entity_points = projected_u[start_idx:end_idx]
                plt.scatter(
                    entity_points[:, 0],
                    entity_points[:, 1],
                    c=[colors[i]],
                    label=f'Entity {i+1}',
                    alpha=0.6
                )
            
            plt.title('P_e-Projected Vectors along First Two Left Singular Vectors (U)')
            plt.xlabel('First Left Singular Vector')
            plt.ylabel('Second Left Singular Vector')
            plt.legend()
            plt.tight_layout()
            plt.savefig('images/entity_projections_U.png')
            plt.close()
            
            return intra_cluster_dist, inter_cluster_dist, effective_rank, U, S, V
        
    # Modified visualization method for EntitySubspaceLearner
    def visualize_pca_comparison(self, entity_vectors: torch.Tensor, U, save_path: str = 'images/pca_comparison.png'):
        """Add this method to EntitySubspaceLearner class"""
        results, pc = compare_pca_spaces(entity_vectors, self.P_e, save_path)
        plt.plot(U@pc)
        plt.savefig("images/principal_component_in_U_decomposition.png")
        

def train_epoch(learner, data_loader, desc="Training"):
    """Run one epoch of training."""
    epoch_losses = {
        'align_loss': 0.0,
        'separate_loss': 0.0,
        'rank_loss': 0.0,
        'total_loss': 0.0
    }
    
    pbar = tqdm(data_loader, desc=desc, leave=False)
    for batch in pbar:
        losses = learner.train_step(batch)
        
        # Update running averages
        for k in epoch_losses:
            epoch_losses[k] += losses[k]
            
        # Update progress bar
        pbar.set_postfix({
            k: f"{v:.4f}" for k, v in losses.items()
        })
    
    # Compute averages
    num_batches = len(data_loader)
    return {k: v/num_batches for k, v in epoch_losses.items()}

def evaluate(learner, data_loader, desc="Validating"):
    """Evaluate model on a dataset."""
    eval_losses = {
        'align_loss': 0.0,
        'separate_loss': 0.0,
        'rank_loss': 0.0,
        'total_loss': 0.0
    }
    
    learner.eval()
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc, leave=False)
        for batch in pbar:
            # Forward pass only
            batch = batch[0].to(learner.device)
            align_loss = learner.entity_alignment_loss(batch)
            separate_loss = learner.entity_separation_loss(batch)
            rank_loss = learner.compute_rank_loss()
            total_loss = align_loss + separate_loss
            
            # Update running averages
            eval_losses['align_loss'] += align_loss.item()
            eval_losses['separate_loss'] += separate_loss.item()
            eval_losses['rank_loss'] += rank_loss.item()
            eval_losses['total_loss'] += total_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                k: f"{v/len(data_loader):.4f}" for k, v in eval_losses.items()
            })
    
    # Compute averages
    num_batches = len(data_loader)
    eval_losses = {k: v/num_batches for k, v in eval_losses.items()}
    
    learner.train()
    return eval_losses

def main():
    # Load cached activations
    activations = torch.load('artifacts/activation_cache.pt')
    print(f"Loaded activations with shape: {activations.shape}")
    
    # Split into train/validation
    val_size = int(0.1 * len(activations))
    train_size = len(activations) - val_size
    
    train_data, val_data = torch.utils.data.random_split(
        activations, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data.dataset[train_data.indices]),
        batch_size=128,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_data.dataset[val_data.indices]),
        batch_size=128,
        shuffle=False,
        drop_last=False
    )
    
    print(f"Train size: {len(train_loader.dataset)}, Validation size: {len(val_loader.dataset)}")
    
    # Initialize learner
    input_dim = activations.shape[-1]
    learner = EntitySubspaceLearner(
        input_dim=input_dim,
        subspace_dim=32,
        margin=0.5,
        rank_weight=1,
    )
    
    # Training loop with early stopping
    num_epochs = 1000
    best_val_loss = float('inf')
    patience = 50
    no_improve = 0
    train_history = []
    val_history = []
    
    epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
    for epoch in epoch_pbar:
        # Train one epoch
        train_losses = train_epoch(learner, train_loader)
        train_history.append(train_losses)
        
        # Evaluate on validation set
        val_losses = evaluate(learner, val_loader)
        val_history.append(val_losses)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f"{train_losses['total_loss']:.4f}",
            'val_loss': f"{val_losses['total_loss']:.4f}"
        })
        
        # Early stopping check
        if val_losses['total_loss'] < best_val_loss:
            best_val_loss = val_losses['total_loss']
            no_improve = 0
            # Save best model
            best_state = learner.state_dict()
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            # Restore best model
            learner.load_state_dict(best_state)
            break
    
    # Plot training history
    plot_training_history(train_history, val_history)
    
    # Analyze results on validation set
    final_val_losses = evaluate(learner, val_loader, desc="Final Evaluation")
    print("\nFinal Validation Results:")
    for k, v in final_val_losses.items():
        print(f"{k}: {v:.4f}")
    
    # Analyze subspace and create visualization using validation set
    intra_dist, inter_dist, eff_rank, U, S, V = learner.analyze_subspace(val_data.dataset[val_data.indices])
    print("\nSubspace Analysis (Validation Set):")
    print(f"Intra-cluster distance: {intra_dist:.4f}")
    print(f"Inter-cluster distance: {inter_dist:.4f}")
    print(f"Effective rank: {eff_rank}")

    learner.visualize_pca_comparison(val_data.dataset[val_data.indices], U)

if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.functional import normalize
# from tqdm import tqdm
# from typing import Tuple, Dict

# class EntitySubspaceLearner(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         subspace_dim: int = 32,
#         margin: float = 0.5,
#         rank_weight: float = 0.1,
#         device: str = "cuda" if torch.cuda.is_available() else "cpu",
#         force_orthogonal: bool = True,
#     ):
#         super().__init__()
#         self.device = device
#         self.subspace_dim = subspace_dim
#         self.margin = margin
#         self.rank_weight = rank_weight
#         self.force_orthogonal = force_orthogonal

#         # Initialize projection matrix as a proper nn.Parameter
#         P_e = torch.empty(subspace_dim, input_dim, device=device)
#         nn.init.orthogonal_(P_e)
#         self.P_e = nn.Parameter(P_e)

#         # Move to device
#         self.to(device)

#         # Initialize optimizer
#         self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

#     def compute_rank_loss(self) -> torch.Tensor:
#         """Compute nuclear norm (trace norm) of the projection matrix to encourage low rank."""
#         # Compute singular values of P_e
#         _, S, _ = torch.svd(self.P_e)
#         # Nuclear norm is the sum of singular values
#         nuclear_norm = torch.sum(S)
#         return torch.tensor(0)

#     def entity_alignment_loss(self, entity_vectors: torch.Tensor) -> torch.Tensor:
#         """Compute alignment loss for entities with same index."""
#         batch_size, num_entities, _ = entity_vectors.shape

#         # Normalize input vectors
#         entity_vectors = normalize(entity_vectors, dim=-1)

#         # Project to subspace
#         flat_vectors = entity_vectors.reshape(-1, entity_vectors.shape[-1])
#         proj_entities = self.P_e @ flat_vectors.t()
#         proj_entities = proj_entities.t()
#         proj_entities = proj_entities.reshape(batch_size, num_entities, self.subspace_dim)
#         proj_entities = normalize(proj_entities, dim=-1)

#         loss = 0.0
#         # For each entity index, compute mean cosine similarity within cluster
#         for entity_idx in range(num_entities):
#             entity_cluster = proj_entities[:, entity_idx, :]
#             sim_matrix = torch.mm(entity_cluster, entity_cluster.t())
#             mask = ~torch.eye(batch_size, dtype=torch.bool, device=self.device)
#             sim_matrix = sim_matrix[mask].reshape(batch_size, batch_size - 1)
#             loss -= sim_matrix.mean()

#         normalized_loss = loss / num_entities
#         normalized_loss += 1.0

#         return normalized_loss / num_entities

#     def entity_separation_loss(self, entity_vectors: torch.Tensor) -> torch.Tensor:
#         """Compute separation loss between different entity indices."""
#         batch_size, num_entities, _ = entity_vectors.shape

#         # Normalize and project vectors
#         entity_vectors = normalize(entity_vectors, dim=-1)
#         flat_vectors = entity_vectors.reshape(-1, entity_vectors.shape[-1])
#         proj_entities = self.P_e @ flat_vectors.t()
#         proj_entities = proj_entities.t()
#         proj_entities = proj_entities.reshape(batch_size, num_entities, self.subspace_dim)
#         proj_entities = normalize(proj_entities, dim=-1)

#         loss = 0.0
#         # Compare each pair of different entity indices
#         for i in range(num_entities):
#             for j in range(i + 1, num_entities):
#                 sim = torch.mm(proj_entities[:, i, :], proj_entities[:, j, :].t())
#                 loss += torch.relu(sim - self.margin).mean()

#         return loss

#     def train_step(self, batch: Tuple[torch.Tensor]) -> Dict[str, float]:
#         """Perform one training step."""
#         entity_vectors = batch[0].to(self.device)
#         self.optimizer.zero_grad()

#         align_loss = self.entity_alignment_loss(entity_vectors)
#         separate_loss = self.entity_separation_loss(entity_vectors)
#         rank_loss = self.compute_rank_loss()

#         total_loss = align_loss + separate_loss + self.rank_weight * rank_loss

#         total_loss.backward()
#         self.optimizer.step()

#         if self.force_orthogonal:
#             with torch.no_grad():
#                 U, _, V = torch.svd(self.P_e)
#                 self.P_e.data = torch.mm(U, V.t())

#         return {
#             'align_loss': align_loss.item(),
#             'separate_loss': separate_loss.item(),
#             'rank_loss': rank_loss.item(),
#             'total_loss': total_loss.item()
#         }

# def train_epoch(learner: EntitySubspaceLearner, data_loader: torch.utils.data.DataLoader, desc: str = "Training") -> Dict[str, float]:
#     """Run one epoch of training."""
#     epoch_losses = {
#         'align_loss': 0.0,
#         'separate_loss': 0.0,
#         'rank_loss': 0.0,
#         'total_loss': 0.0
#     }

#     pbar = tqdm(data_loader, desc=desc, leave=False)
#     for batch in pbar:
#         print(f'train batch shape: {batch}')
#         losses = learner.train_step(batch)

#         for k in epoch_losses:
#             epoch_losses[k] += losses[k]

#         pbar.set_postfix({
#             k: f"{v:.4f}" for k, v in losses.items()
#         })

#     num_batches = len(data_loader)
#     return {k: v/num_batches for k, v in epoch_losses.items()}

# def evaluate(learner: EntitySubspaceLearner, data_loader: torch.utils.data.DataLoader, desc: str = "Validating") -> Dict[str, float]:
#     """Evaluate model on a dataset."""
#     eval_losses = {
#         'align_loss': 0.0,
#         'separate_loss': 0.0,
#         'rank_loss': 0.0,
#         'total_loss': 0.0
#     }

#     learner.eval()
#     with torch.no_grad():
#         pbar = tqdm(data_loader, desc=desc, leave=False)
#         for batch in pbar:
#             batch = batch[0].to(learner.device)
#             align_loss = learner.entity_alignment_loss(batch)
#             separate_loss = learner.entity_separation_loss(batch)
#             rank_loss = learner.compute_rank_loss()
#             total_loss = align_loss + separate_loss

#             eval_losses['align_loss'] += align_loss.item()
#             eval_losses['separate_loss'] += separate_loss.item()
#             eval_losses['rank_loss'] += rank_loss.item()
#             eval_losses['total_loss'] += total_loss.item()

#             pbar.set_postfix({
#                 k: f"{v/len(data_loader):.4f}" for k, v in eval_losses.items()
#             })

#     num_batches = len(data_loader)
#     eval_losses = {k: v/num_batches for k, v in eval_losses.items()}

#     learner.train()
#     return eval_losses


# def train(
#     activation_files: list,  # List of file paths to activation caches
#     subspace_dim: int = 32,
#     margin: float = 0.5,
#     rank_weight: float = 1.0,
#     batch_size: int = 128,
#     num_epochs: int = 1000,
#     patience: int = 50,
#     val_size: float = 0.1,
#     rows_per_file: int = 1000,
# ) -> Tuple[EntitySubspaceLearner, Dict[str, list]]:
#     """Train the EntitySubspaceLearner model on combined activations from multiple files."""
#     # Load and combine activations from multiple files
#     combined_activations = []

#     for file_path in activation_files:
#         # Load activation cache
#         activations = torch.load(file_path)
#         # Take first rows_per_file rows
#         activations = activations[:rows_per_file]
#         combined_activations.append(activations)

#     # Stack all activations into one tensor
#     combined_activations = torch.cat(combined_activations, dim=0)

#     # Shuffle the combined activations
#     shuffle_indices = torch.randperm(len(combined_activations))
#     combined_activations = combined_activations[shuffle_indices]
#     print(f"Combined activations shape: {combined_activations.shape}")

#     # Split into train/validation
#     val_size = int(val_size * len(combined_activations))
#     train_size = len(combined_activations) - val_size

#     train_data, val_data = torch.utils.data.random_split(
#         combined_activations,
#         [train_size, val_size],
#         generator=torch.Generator().manual_seed(42)
#     )

#     # Create DataLoaders
#     train_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(train_data.dataset[train_data.indices]),
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=True
#     )

#     val_loader = torch.utils.data.DataLoader(
#         torch.utils.data.TensorDataset(val_data.dataset[val_data.indices]),
#         batch_size=batch_size,
#         shuffle=False,
#         drop_last=False
#     )

#     print(f"Train size: {len(train_loader.dataset)}, Validation size: {len(val_loader.dataset)}")

#     # Initialize learner
#     input_dim = combined_activations.shape[-1]
#     learner = EntitySubspaceLearner(
#         input_dim=input_dim,
#         subspace_dim=subspace_dim,
#         margin=margin,
#         rank_weight=rank_weight,
#     )

#     # Training loop with early stopping
#     best_val_loss = float('inf')
#     no_improve = 0
#     train_history = []
#     val_history = []

#     epoch_pbar = tqdm(range(num_epochs), desc='Epochs')
#     for epoch in epoch_pbar:
#         # Train one epoch
#         train_losses = train_epoch(learner, train_loader)
#         train_history.append(train_losses)

#         # Evaluate on validation set
#         val_losses = evaluate(learner, val_loader)
#         val_history.append(val_losses)

#         # Update epoch progress bar
#         epoch_pbar.set_postfix({
#             'train_loss': f"{train_losses['total_loss']:.4f}",
#             'val_loss': f"{val_losses['total_loss']:.4f}"
#         })

#         # Early stopping check
#         if val_losses['total_loss'] < best_val_loss:
#             best_val_loss = val_losses['total_loss']
#             no_improve = 0
#             best_state = learner.state_dict()
#         else:
#             no_improve += 1

#         if no_improve >= patience:
#             print(f"\nEarly stopping at epoch {epoch+1}")
#             print(f"Best validation loss: {best_val_loss:.4f}")
#             learner.load_state_dict(best_state)
#             break

#     history = {
#         'train': train_history,
#         'val': val_history
#     }

#     return learner, history