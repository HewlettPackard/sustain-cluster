# To be added to a file like rl_components/meta_task_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
class MetaTaskEncoder(nn.Module):
    """
    Encodes a variable-length set of tasks into a single fixed-size embedding
    using a Transformer Encoder.
    """
    def __init__(self,
                 task_feature_dim: int,
                 embedding_dim: int,  # The output dimension, e.g., D_META_MANAGER
                 num_heads: int = 4,
                 num_encoder_layers: int = 2,
                 ff_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 1. Input Projection Layer
        # Projects the raw task features into the model's working dimension (embedding_dim)
        self.input_projection = nn.Linear(task_feature_dim, embedding_dim)
        
        # 2. Transformer Encoder
        # This will process the set of projected task vectors
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation=F.relu
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 3. Special "CLS" Token (inspired by BERT)
        # This is a learnable parameter that will act as a "summary" token.
        # We will prepend it to the sequence of tasks.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self,
                task_set: torch.Tensor,    # Shape: (Batch, MaxNumTasks, D_TASK_FEAT)
                padding_mask: torch.Tensor # Shape: (Batch, MaxNumTasks), True where padded
               ) -> torch.Tensor:
        """
        Args:
            task_set: A padded batch of task feature vectors.
            padding_mask: A boolean mask indicating which entries are padding.

        Returns:
            A fixed-size embedding for the entire task set.
            Shape: (Batch, embedding_dim)
        """
        # Project the raw features into the embedding space
        # -> (Batch, MaxNumTasks, embedding_dim)
        task_embeddings = self.input_projection(task_set)
        
        # Prepend the CLS token to each sequence in the batch
        B = task_embeddings.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # -> (Batch, MaxNumTasks + 1, embedding_dim)
        full_sequence = torch.cat([cls_tokens, task_embeddings], dim=1)
        
        # Create a new padding mask for the full sequence (CLS token is never padded)
        # -> (Batch, MaxNumTasks + 1)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=task_set.device)
        full_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Pass through the Transformer Encoder
        # -> (Batch, MaxNumTasks + 1, embedding_dim)
        encoded_sequence = self.transformer_encoder(
            src=full_sequence,
            src_key_padding_mask=full_padding_mask
        )

        # The final embedding for the entire set of tasks is the output of the CLS token.
        # This token has aggregated information from all other tasks via the self-attention mechanism.
        # -> (Batch, embedding_dim)
        meta_task_embedding = encoded_sequence[:, 0, :]
        
        return meta_task_embedding

# To be added to a file like rl_components/meta_task_encoder.py

class MetaTaskDecoder(nn.Module):
    """
    Decodes a fixed-size meta-task embedding back into a sequence of
    task feature vectors using a Transformer Decoder.
    """
    def __init__(self,
                 task_feature_dim: int,
                 embedding_dim: int,
                 num_heads: int = 4,
                 num_decoder_layers: int = 2,
                 ff_dim: int = 128):
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation=F.relu
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # A final linear layer to project from the embedding dimension
        # back to the original task feature dimension.
        self.output_projection = nn.Linear(embedding_dim, task_feature_dim)

    def forward(self,
                target_sequence: torch.Tensor,   # The sequence to be generated
                memory: torch.Tensor,            # The embedding from the ENCODER
                padding_mask: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """
        Args:
            target_sequence: The input to the decoder (e.g., a shifted version of the original tasks).
            memory: The context from the encoder (the meta-task embedding).
            padding_mask: Mask for the target sequence.
        
        Returns:
            The reconstructed sequence of task vectors.
            Shape: (Batch, MaxNumTasks, D_TASK_FEAT)
        """
        # The decoder needs a memory input that matches the sequence length.
        # We'll expand the single embedding vector from the encoder.
        seq_len = target_sequence.shape[1]
        memory_expanded = memory.unsqueeze(1).expand(-1, seq_len, -1)

        # Get the output from the transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=target_sequence,
            memory=memory_expanded,
            tgt_key_padding_mask=padding_mask
        )
        
        # Project the output back to the original feature dimension
        reconstructed_tasks = self.output_projection(decoder_output)
        
        return reconstructed_tasks

class TaskAutoencoder(nn.Module):
    def __init__(self, encoder: MetaTaskEncoder, decoder: MetaTaskDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,
                task_set: torch.Tensor,
                padding_mask: torch.Tensor
               ) -> torch.Tensor:
        
        # 1. Encode the task set into a fixed-size embedding
        meta_task_embedding = self.encoder(task_set, padding_mask)
        
        # For auto-encoding, the decoder's input `target_sequence` is typically
        # the same as the original input, or a slightly shifted version.
        # Here, we can just use the original projected embeddings.
        projected_input = self.encoder.input_projection(task_set)

        # 2. Decode the embedding back into a sequence of tasks
        reconstructed_tasks = self.decoder(
            target_sequence=projected_input,
            memory=meta_task_embedding,
            padding_mask=padding_mask
        )
        
        return reconstructed_tasks