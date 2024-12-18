import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import Tuple, Dict, List
import os


class EntityActivationDataset(Dataset):
    """Dataset class to handle entity activations with their types."""

    def __init__(self, activation_files: Dict[int, str]):
        """
        Args:
            activation_files: Dictionary mapping entity type (int) to file path
        """
        self.data = []
        self.entity_types = []

        # Load activations from each file
        for entity_type, file_path in activation_files.items():
            activations = torch.load(file_path)
            self.data.append(activations)
            self.entity_types.extend([entity_type] * len(activations))

        # Convert to tensors
        self.data = torch.cat(self.data, dim=0)
        self.entity_types = torch.tensor(self.entity_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.entity_types[idx]


class EntitySubspaceLearner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        subspace_dim: int = 32,
        separate_margin: float = 0.5,
        lr: float = 1e-3,
        align_coeff: float = 1.0,
        separate_coeff: float = 1.0,
        rank_coeff: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        force_orthogonal: bool = False,
    ):
        super().__init__()
        self.device = device
        self.subspace_dim = subspace_dim
        self.separate_margin = separate_margin
        self.align_coeff = align_coeff
        self.separate_coeff = separate_coeff
        self.rank_coeff = rank_coeff
        self.force_orthogonal = force_orthogonal

        # Initialize projection matrix
        P_e = torch.empty(subspace_dim, input_dim, device=device)
        nn.init.orthogonal_(P_e)
        # P_e = torch.rand(subspace_dim, input_dim, device=device)
        self.P_e = nn.Parameter(P_e)

        # Move to device
        self.to(device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def compute_rank_loss(self) -> torch.Tensor:
        """Compute nuclear norm of the projection matrix."""
        _, S, _ = torch.svd(self.P_e)
        nuclear_norm = torch.sum(S)
        return nuclear_norm

    def compute_similarity_losses(
        self, entity_vectors: torch.Tensor, entity_types: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute alignment and separation losses across the batch."""
        # Normalize and project vectors
        entity_vectors = normalize(entity_vectors, dim=-1)

        proj_entities = self.P_e @ entity_vectors.t()
        proj_entities = proj_entities.t()
        proj_entities = normalize(proj_entities, dim=-1)

        # Compute similarity matrix for all pairs
        sim_matrix = torch.mm(proj_entities, proj_entities.t())

        # Create masks for same and different entity types
        entity_type_matrix = entity_types.unsqueeze(0) == entity_types.unsqueeze(1)
        identity_mask = torch.eye(len(entity_vectors), dtype=torch.bool, device=self.device)
        same_type_mask = entity_type_matrix & ~identity_mask
        diff_type_mask = ~entity_type_matrix & ~identity_mask

        # Compute alignment loss (negative mean similarity for same type)
        align_loss = 1.0 - sim_matrix[same_type_mask].mean()

        # Compute separation loss (hinge loss for different types)
        separate_loss = torch.relu(sim_matrix[diff_type_mask].abs() - self.separate_margin).mean()

        return {"align_loss": align_loss, "separate_loss": separate_loss}

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        entity_vectors, entity_types = batch
        entity_vectors = entity_vectors.to(self.device)
        entity_types = entity_types.to(self.device)

        self.optimizer.zero_grad()

        losses = self.compute_similarity_losses(entity_vectors, entity_types)
        rank_loss = self.compute_rank_loss()

        total_loss = (
            self.align_coeff * losses["align_loss"]
            + self.separate_coeff * losses["separate_loss"]
            + self.rank_coeff * rank_loss
        )

        total_loss.backward()
        self.optimizer.step()

        if self.force_orthogonal:
            with torch.no_grad():
                U, _, V = torch.svd(self.P_e)
                self.P_e.data = torch.mm(U, V.t())

        return {
            "align_loss": losses["align_loss"].item(),
            "separate_loss": losses["separate_loss"].item(),
            "rank_loss": rank_loss.item(),
            "total_loss": total_loss.item(),
        }


def train_epoch(
    learner: EntitySubspaceLearner, data_loader: DataLoader, desc: str = "Training"
) -> Dict[str, float]:
    """Run one epoch of training."""
    epoch_losses = {"align_loss": 0.0, "separate_loss": 0.0, "rank_loss": 0.0, "total_loss": 0.0}

    pbar = tqdm(data_loader, desc=desc, leave=False)
    for batch in pbar:
        losses = learner.train_step(batch)

        for k in epoch_losses:
            epoch_losses[k] += losses[k]

        pbar.set_postfix({k: f"{v:.4f}" for k, v in losses.items()})

    num_batches = len(data_loader)
    return {k: v / num_batches for k, v in epoch_losses.items()}


def evaluate(
    learner: EntitySubspaceLearner, data_loader: DataLoader, desc: str = "Validating"
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    eval_losses = {"align_loss": 0.0, "separate_loss": 0.0, "rank_loss": 0.0, "total_loss": 0.0}

    learner.eval()
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc, leave=False)
        for batch in pbar:
            entity_vectors, entity_types = batch
            entity_vectors = entity_vectors.to(learner.device)
            entity_types = entity_types.to(learner.device)

            losses = learner.compute_similarity_losses(entity_vectors, entity_types)
            rank_loss = learner.compute_rank_loss()
            total_loss = (
                learner.align_coeff * losses["align_loss"]
                + learner.separate_coeff * losses["separate_loss"]
                + learner.rank_coeff * rank_loss
            )

            eval_losses["align_loss"] += losses["align_loss"].item()
            eval_losses["separate_loss"] += losses["separate_loss"].item()
            eval_losses["rank_loss"] += rank_loss.item()
            eval_losses["total_loss"] += total_loss.item()

            pbar.set_postfix({k: f"{v/len(data_loader):.4f}" for k, v in eval_losses.items()})

    num_batches = len(data_loader)
    eval_losses = {k: v / num_batches for k, v in eval_losses.items()}

    learner.train()
    return eval_losses


def train(
    activation_files: Dict[int, str],  # Dictionary mapping entity type to file path
    patience: int = 50,
    val_size: float = 0.1,
    batch_size: int = 128,
    num_epochs: int = 1000,
    **learner_kwargs,
) -> Tuple[EntitySubspaceLearner, Dict[str, list]]:
    """Train the EntitySubspaceLearner model."""
    # Create dataset
    dataset = EntityActivationDataset(activation_files)

    # Split into train/validation
    val_size = int(val_size * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # Initialize learner
    input_dim = dataset.data.shape[-1]
    learner = EntitySubspaceLearner(
        input_dim=input_dim,
        **learner_kwargs,
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    no_improve = 0
    train_history = []
    val_history = []

    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_pbar:
        # Train one epoch
        train_losses = train_epoch(learner, train_loader)
        train_history.append(train_losses)

        # Evaluate on validation set
        val_losses = evaluate(learner, val_loader)
        val_history.append(val_losses)

        # Update epoch progress bar
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_losses['total_loss']:.4f}",
                "val_loss": f"{val_losses['total_loss']:.4f}",
            }
        )

        # Early stopping check
        if best_val_loss - val_losses["total_loss"] > 1e-4:
            best_val_loss = val_losses["total_loss"]
            no_improve = 0
            best_state = learner.state_dict()
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            learner.load_state_dict(best_state)
            break

    history = {"train": train_history, "val": val_history}

    return learner, history, train_dataset, val_dataset
