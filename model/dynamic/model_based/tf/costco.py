import torch
from torch import nn
import torch.nn.functional as F


class EmbeddingModule(nn.Module):
    """
    EmbeddingModule maps each tensor mode (dimension) into a shared latent space.

    For a tensor of order N, this module learns N embedding matrices:
    - Each embedding represents one mode (e.g., user, item, time)
    - All embeddings share the same latent rank (factor dimension)

    Output:
        A stacked latent representation of shape:
        (batch_size, rank, num_modes)
    """
    def __init__(self, tensor_shape, rank):
        super(EmbeddingModule, self).__init__()
        self.tensor_shape = tensor_shape

        # Create an embedding table for each tensor mode
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=d, embedding_dim=rank) for d in tensor_shape]
        )

    def forward(self, indices):
        """
        Args:
            indices: Tensor of shape (batch_size, num_modes)
                     Each column corresponds to one tensor mode index

        Returns:
            Concatenated embedding tensor of shape (batch_size, rank, num_modes)
        """
        # Extract embeddings for each mode and add a new dimension for concatenation
        x = [
            self.embeddings[i](indices[:, i]).unsqueeze(2)
            for i in range(len(self.tensor_shape))
        ]

        # Concatenate along the mode dimension
        x = torch.cat(x, dim=2)

        return x


class MappingModule(nn.Module):
    """
    MappingModule captures high-order interactions among latent factors
    using 2D convolution operations.

    Key idea:
    - First convolution aggregates information across embedding dimensions (rank)
    - Second convolution aggregates across tensor modes (e.g., user-item-time)
    """
    def __init__(self, tensor_shape, rank, num_channels):
        super(MappingModule, self).__init__()

        # Convolution across latent dimension (rank-wise interaction)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=(rank, 1)
        )

        # Convolution across modes (mode-wise interaction)
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(1, len(tensor_shape))
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, rank, num_modes)

        Returns:
            Feature tensor after interaction modeling
        """
        # Add channel dimension for convolution: (B, 1, rank, num_modes)
        x = x.unsqueeze(1)

        # Capture interactions along latent dimension
        x = F.relu(self.conv1(x))

        # Capture interactions across modes
        x = F.relu(self.conv2(x))

        return x


class AggregationModule(nn.Module):
    """
    AggregationModule transforms convolutional feature maps into final prediction.

    It acts as a nonlinear regressor:
    - Flatten interaction features
    - Apply MLP for final QoS/value prediction
    """
    def __init__(self, num_channels):
        super(AggregationModule, self).__init__()

        # Fully connected layers for regression
        self.fc1 = nn.Linear(num_channels, num_channels)
        self.fc2 = nn.Linear(num_channels, 1)

    def forward(self, x):
        """
        Args:
            x: Feature tensor from MappingModule

        Returns:
            Scalar prediction for each sample
        """
        # Flatten spatial dimensions
        x = x.flatten(start_dim=1)

        # Nonlinear transformation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


class CoSTCoModel(nn.Module):
    """
    CoSTCo Model (Convolutional Space Tensor Completion)

    Architecture:
        EmbeddingModule  ->  MappingModule  ->  AggregationModule

    Workflow:
    1. Embed each tensor mode into a shared latent space
    2. Use convolutional layers to model high-order interactions
    3. Aggregate features via MLP to produce final prediction

    This design can be viewed as a neural tensor factorization model
    enhanced with convolutional interaction learning.
    """
    def __init__(self, field_dims, field_order, config):
        super(CoSTCoModel, self).__init__()

        rank = config['CoSTCo']['rank']                # latent factor dimension
        num_channels = config['CoSTCo']['num_channels']  # number of convolution filters

        # Initialize modules
        self.embedding_module = EmbeddingModule(field_dims, rank)
        self.mapping_module = MappingModule(field_dims, rank, num_channels)
        self.aggregation_module = AggregationModule(num_channels)

    def forward(self, indices):
        """
        Args:
            indices: Tensor of shape (batch_size, num_modes)

        Returns:
            Final scalar prediction (e.g., QoS value)
        """
        # Step 1: Latent embedding
        x = self.embedding_module(indices)

        # Step 2: High-order interaction modeling via CNN
        x = self.mapping_module(x)

        # Step 3: Prediction via MLP aggregation
        x = self.aggregation_module(x)

        return x.squeeze()