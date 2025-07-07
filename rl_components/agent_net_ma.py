# """
# • **ManagerNet**  attention scorer that picks **which data-center (DC)** should
#   run a meta-task.  It supports an *automatic* **padding mask**: if your input
#   tensor already reserves space for `num_clusters` DCs (e.g. 10) but the current
#   scenario only has `active_clusters` (e.g. 3), simply pass that integer and the
#   remaining padded DC slots will be hard-masked (logits = −∞).

# • **WorkerNet**  classifier that decides **execute now** vs **defer** for the
#   local queue.

# Both networks are framework-agnostic and can be dropped into any PyTorch
# training loop or higher-level library.
# """

# import torch 
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import Optional, Tuple

# class MLP(nn.Module):
#     """Two layer MLP with optional LayerNorm."""

#     def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, layer_norm: bool = False):
#         super().__init__()
#         layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
#         if layer_norm:
#             layers.append(nn.LayerNorm(hidden_dim))
#         layers += [nn.ReLU(), nn.Linear(hidden_dim, out_dim)]
#         self.net = nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
#         return self.net(x)
    

# class AttentionModule(nn.Module):
#     """Self-attention over option set (mask aware)."""

#     def __init__(self, emb_dim: int, num_layers: int = 2, num_heads: int = 2, ff_dim: int = 256):
#         super().__init__()
#         layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,
#                                                dim_feedforward=ff_dim, batch_first=True)
#         self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

#     def forward(self, seq_emb_option_initial: torch.Tensor, 
#                 all_options_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
#         return self.encoder(seq_emb_option_initial, src_key_padding_mask=all_options_padding_mask)
    
# # class ManagerActor(nn.Module):
# #     """
# #     Attention-based policy π-manager actor
# #     """
# #     def __init__(self, 
# #                  D_emb_meta_manager: int, 
# #                  D_global: int, 
# #                  D_option_feat: int, # This is 2
# #                  max_total_options, 
# #                  *, 
# #                  hidden_dim: int = 32,
# #                  embed_dim: int = 64,  # The dimension for the Transformer
# #                  num_heads: int = 4): # Now 64 is divisible by 4
        
# #         super().__init__()
# #         self.max_total_options = max_total_options

# #         # 1. ENCODER: Project raw features (size 2) to a rich embedding (size 64)
# #         self.option_encoder = MLP(D_option_feat, hidden_dim, embed_dim)

# #         # 2. ATTENTION: Operates on the rich embeddings (size 64)
# #         self.attn = AttentionModule(emb_dim=embed_dim, num_heads=num_heads)

# #         # 3. QUERY & SCORER: Also operate on the rich embedding dimension
# #         self.query = MLP(D_emb_meta_manager + D_global, hidden_dim, embed_dim)
# #         self.scorer = MLP(embed_dim * 2, hidden_dim, 1)

# #     def forward(self, 
# #                 emb_meta_task_mgr,              #(B, D_meta_manager)
# #                 emb_global_context_mgr,         #(B, D_global)
# #                 obs_all_options_set_padded,     #(B, max_total_options, D_option_feat)
# #                 all_options_padding_mask):     #(B, max_total_options)
        
# #         B, max_total_options, D_option_feat = obs_all_options_set_padded.shape

# #         # Step 1: Encode the raw features
# #         seq_emb_option_initial = self.option_encoder(obs_all_options_set_padded) # Shape becomes (B, max_total_options, embed_dim)

# #         # Step 2: Perform attention on the rich embeddings
# #         seq_emb_options_contextual = self.attn(seq_emb_option_initial, all_options_padding_mask) # Shape remains (B, max_total_options, embed_dim)
        
# #         # Step 3: Compute the query vector
# #         # Concatenate the meta-task and global context embeddings
# #         # and pass through the query MLP
# #         query = self.query(torch.cat([emb_meta_task_mgr, emb_global_context_mgr], dim=1))
        
# #         # Unsqueeze to add a 'sequence' dimension. Shape: (B, 1, embed_dim)
# #         query_expanded = query.unsqueeze(1)
        
# #         # Expand the query to match the number of options.
# #         # The '-1' correctly tells it to keep the last dimension (embed_dim) as is.
# #         # Output shape: (B, max_total_options, embed_dim), e.g., (B, max_total_options, 64)
# #         query_expanded = query_expanded.expand(-1, self.max_total_options, -1)
        
# #         # Now the shapes match for concatenation:
# #         # query_expanded: (B, max_total_options, 64)
# #         # seq_emb_options_contextual: (B, max_total_options, 64)
# #         fused = torch.cat([query_expanded, seq_emb_options_contextual], dim=-1) # Fused shape: (B, max_total_options, 128)
        
# #         logits = self.scorer(fused).squeeze(-1) #(B, max_total_options)

# #         if all_options_padding_mask is not None:
# #             logits = logits.masked_fill(all_options_padding_mask, float('-inf'))

# #         return logits
    
# #     def sample_action(self, emb_meta_task_mgr,             
# #                 emb_global_context_mgr,         
# #                 obs_all_options_set_padded,     
# #                 all_options_padding_mask):
# #         """
# #         Samples actions, log_probs, and entropy from the policy distribution.
# #         This method is useful for on-policy algorithms or for evaluation.
# #         For SAC, typically only forward() is called and distribution handled externally.
# #         """

# #         logits = self.forward(emb_meta_task_mgr,             
# #                 emb_global_context_mgr,         
# #                 obs_all_options_set_padded,     
# #                 all_options_padding_mask)
        
# #         probs = F.softmax(logits, dim=-1) # Softmax to get probabilities
# #         dist = torch.distributions.Categorical(probs=probs) # Use probs for Categorical
# #         actions = dist.sample()
# #         log_probs = dist.log_prob(actions)
# #         entropy = dist.entropy().mean() # Average entropy over the batch of tasks
# #         return actions, log_probs, entropy
    
# # class ManagerCritic(nn.Module):
# #     """
# #     Twin-Q critic for π-Manager
# #       • q_values_all_options  = forward_q_values(...)
# #       • q1,q2 for chosen idx  = q_for_action(...)
# #     """
# #     def __init__(self,
# #                  D_emb_meta_manager: int, 
# #                  D_global: int, 
# #                  D_option_feat: int, 
# #                  max_total_options, 
# #                  *, hidden_dim: int=128):
        
# #         super().__init__()
# #         self.max_total_options = max_total_options
# #         self.D_option_feat = D_option_feat
# #         self.attn = AttentionModule(D_option_feat)
# #         self.query = MLP(D_emb_meta_manager +  D_global, hidden_dim, D_option_feat)
# #         self.Q1 = MLP(D_option_feat * 2, hidden_dim, 1)
# #         self.Q2 = MLP(D_option_feat * 2, hidden_dim, 1) 


# #     def forward_q_values(self,
# #                          emb_meta_task_mgr,             
# #                          emb_global_context_mgr,         
# #                          obs_all_options_set_padded,   
# #                          all_options_padding_mask):
# #         """
# #         return: q1_all, q2_all  →  (B, max_total_options)
# #         """
# #         seq_emb_options_contextual   = self.attn(obs_all_options_set_padded, all_options_padding_mask)                                   
# #         query = self.query(torch.cat([emb_meta_task_mgr, emb_global_context_mgr], -1))                
# #         query = query.unsqueeze(1).expand(-1, seq_emb_options_contextual.size(1), -1)          
# #         fused = torch.cat([query, seq_emb_options_contextual], -1)
# #         q1_all = self.Q1(fused).squeeze(-1)
# #         q2_all = self.Q2(fused).squeeze(-1)
# #         return q1_all, q2_all
    
# #     def q_for_action(self,
# #                      emb_meta_task_mgr,             
# #                      emb_global_context_mgr, 
# #                      obs_all_options_set_padded,
# #                      action_idx,                       # LongTensor (B,)
# #                      all_options_padding_mask=None):

# #         q1_all, q2_all = self.forward_q_values(
# #             emb_meta_task_mgr,             
# #             emb_global_context_mgr,
# #             obs_all_options_set_padded,
# #             all_options_padding_mask)

# #         q_selected = lambda q: q.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
# #         return q_selected(q1_all), q_selected(q2_all)  

# class SharedFeatureExtractor(nn.Module):
#     def __init__(self, D_option_feat: int, actor_hidden_dim: int, actor_embed_dim: int, 
#                  critic_attn_embed_dim: int, # Potentially different for critic if needed, but usually same
#                  num_attn_layers: int = 2, num_attn_heads: int = 2, attn_ff_dim: int = 256):
#         super().__init__()
#         # This encoder will be shared by Actor and Critic
#         self.option_encoder = MLP(D_option_feat, actor_hidden_dim, actor_embed_dim) # Projects to actor's desired embed_dim

#         # This attention module will also be shared
#         # It operates on the output of option_encoder
#         self.shared_attn = AttentionModule(emb_dim=actor_embed_dim, # Use actor_embed_dim here
#                                            num_layers=num_attn_layers, 
#                                            num_heads=num_attn_heads, 
#                                            ff_dim=attn_ff_dim)

#     def forward(self, obs_all_options_set_padded, all_options_padding_mask):
#         # Encode raw features
#         seq_emb_option_initial = self.option_encoder(obs_all_options_set_padded)
#         # Perform attention on rich embeddings
#         seq_emb_options_contextual = self.shared_attn(seq_emb_option_initial, all_options_padding_mask)
#         return seq_emb_options_contextual
    

# class WorkerActor(nn.Module):
#     """
#     MLP-based policy π-worker actor
#     """
#     def __init__(self, D_meta_worker: int, D_local_worker: int, D_global: int,  *, hidden_dim: int=128):
#         super().__init__()
#         in_dim = D_meta_worker + D_local_worker + D_global
#         self.mlp = MLP(in_dim, hidden_dim, 2)

#     def forward(self, obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context):
#         combined_input = torch.cat([obs_worker_meta_task_i, obs_local_dc_i_for_worker,obs_global_context], -1)
#         logits_worker_action = self.mlp(combined_input)

#         return logits_worker_action
    
#     def sample_action(self, obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context):
#         logits = self.forward(obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context)
#         probs = F.softmax(logits, dim=-1)
#         dist = torch.distributions.Categorical(probs=probs)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         entropy = dist.entropy().mean()

#         return action, log_prob, entropy
    

# class WorkerCritic(nn.Module):
#     """
#     Twin-Q critic for π-worker
#     """

#     def __init__(self, D_meta_worker: int, D_local_worker: int, D_global: int,  *, hidden_dim: int=128):
#         super().__init__()
#         in_dim = D_meta_worker + D_local_worker + D_global
#         self.Q1 = MLP(in_dim, hidden_dim, 2)
#         self.Q2 = MLP(in_dim, hidden_dim, 2)

#     def forward_q_values(self, obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context):
#         combined_input = torch.cat([obs_worker_meta_task_i, obs_local_dc_i_for_worker,obs_global_context], -1)
#         q1 = self.Q1(combined_input)
#         q2 = self.Q2(combined_input)
#         return q1, q2
    
#     def q_for_action(self,obs_worker_meta_task_i, obs_local_dc_i_for_worker, obs_global_context, action_idx):
#         q1, q2 = self.forward_q_values(obs_worker_meta_task_i, obs_local_dc_i_for_worker,obs_global_context)
#         q_selected = lambda q: q.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
#         return q_selected(q1), q_selected(q2)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MLP(nn.Module):
    """
    Two layer MLP with optional LayerNorm.
    Uses a more standard block structure.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, layer_norm: bool = False):
        super().__init__()
        
        block: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        ]
        if layer_norm:
            block.append(nn.LayerNorm(hidden_dim))
            
        self.block1 = nn.Sequential(*block)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.output_layer(x)
        return x

class AttentionModule(nn.Module):
    """Self-attention over option set (mask aware)."""
    def __init__(self, emb_dim: int, num_layers: int = 2, num_heads: int = 2, ff_dim: int = 256):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by num_heads ({num_heads})")
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads,
                                               dim_feedforward=ff_dim, batch_first=True, activation=F.relu)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, seq_emb_option_initial: torch.Tensor,
                all_options_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(seq_emb_option_initial, src_key_padding_mask=all_options_padding_mask)

class SharedFeatureExtractor(nn.Module):
    """
    Shared module to encode raw option features and apply attention.
    Output is contextual embeddings for each option.
    """
    def __init__(self, D_option_feat: int, encoder_hidden_dim: int, embed_dim: int,
                 num_attn_layers: int = 2, num_attn_heads: int = 2, attn_ff_dim: int = 256):
        super().__init__()
        self.option_encoder = MLP(D_option_feat, encoder_hidden_dim, embed_dim, layer_norm=True)
        self.shared_attn = AttentionModule(emb_dim=embed_dim,
                                           num_layers=num_attn_layers,
                                           num_heads=num_attn_heads,
                                           ff_dim=attn_ff_dim)
        self.embed_dim = embed_dim # Store for easy access by downstream modules

    def forward(self, obs_all_options_set_padded: torch.Tensor,
                all_options_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_emb_option_initial = self.option_encoder(obs_all_options_set_padded)
        seq_emb_options_contextual = self.shared_attn(seq_emb_option_initial, all_options_padding_mask)
        return seq_emb_options_contextual

class ManagerActor(nn.Module):
    """
    Attention-based policy π-manager actor.
    Uses a shared feature extractor for option processing.
    """
    def __init__(self,
                 shared_feature_extractor: SharedFeatureExtractor,
                 D_emb_meta_manager: int,
                 D_global: int,
                 max_total_options: int,
                 *,
                 query_hidden_dim: int = 32,
                 scorer_hidden_dim: int = 32
                ):
        super().__init__()
        self.shared_extractor = shared_feature_extractor
        self.max_total_options = max_total_options
        self.embed_dim = self.shared_extractor.embed_dim

        self.query_mlp = MLP(D_emb_meta_manager + D_global, query_hidden_dim, self.embed_dim, layer_norm=True)
        self.scorer_mlp = MLP(self.embed_dim * 2, scorer_hidden_dim, 1, layer_norm=True)

    def forward(self,
                emb_meta_task_mgr: torch.Tensor,
                emb_global_context_mgr: torch.Tensor,
                obs_all_options_set_padded: torch.Tensor,
                all_options_padding_mask: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        seq_emb_options_contextual = self.shared_extractor(obs_all_options_set_padded, all_options_padding_mask)
        query_input = torch.cat([emb_meta_task_mgr, emb_global_context_mgr], dim=1)
        query_embedding = self.query_mlp(query_input)
        query_expanded = query_embedding.unsqueeze(1)
        query_broadcast = query_expanded.expand(-1, self.max_total_options, -1)
        fused_input = torch.cat([query_broadcast, seq_emb_options_contextual], dim=-1)
        logits = self.scorer_mlp(fused_input)
        logits_squeezed = logits.squeeze(-1)

        if all_options_padding_mask is not None:
            if not all_options_padding_mask.dtype == torch.bool:
                all_options_padding_mask = all_options_padding_mask.bool()
            logits_squeezed = logits_squeezed.masked_fill(all_options_padding_mask, float('-inf'))
        return logits_squeezed

    def sample_action(self,
                      emb_meta_task_mgr: torch.Tensor,
                      emb_global_context_mgr: torch.Tensor,
                      obs_all_options_set_padded: torch.Tensor,
                      all_options_padding_mask: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: # Now returns 4 items
        # The forward pass already applies the mask and returns masked logits
        logits = self.forward(emb_meta_task_mgr,
                              emb_global_context_mgr,
                              obs_all_options_set_padded,
                              all_options_padding_mask)
        
        # We use the unmasked logits for creating the distribution
        # The mask in forward() sets invalid logits to -inf, so softmax handles it correctly
        dist = torch.distributions.Categorical(logits=logits)
        
        actions = dist.sample()
        
        # Calculate log_prob for the *sampled* action (for storing in buffer)
        action_log_probs = dist.log_prob(actions)
        
        entropy = dist.entropy().mean()

        # Return the raw logits as well for the V-value calculation
        return actions, action_log_probs, entropy, logits

class ManagerCritic(nn.Module):
    """
    Twin-Q critic for π-Manager.
    Uses a shared feature extractor for option processing.
    """
    def __init__(self,
                 shared_feature_extractor: SharedFeatureExtractor, # Instance of the shared part
                 D_emb_meta_manager: int,
                 D_global: int,
                 max_total_options: int,
                 *,
                 query_hidden_dim: int = 128, # Hidden dim for the critic's query MLP
                 q_hidden_dim: int = 128      # Hidden dim for the Q-function MLPs
                ):
        super().__init__()
        self.shared_extractor = shared_feature_extractor
        self.max_total_options = max_total_options

        # The embedding dimension is determined by the shared_feature_extractor
        self.embed_dim = self.shared_extractor.embed_dim

        # Critic-specific Query MLP: Takes concatenated meta-task and global context,
        # projects to self.embed_dim to match option embeddings.
        self.query_mlp = MLP(D_emb_meta_manager + D_global, query_hidden_dim, self.embed_dim, layer_norm=True)

        # Twin Q-functions (Q1 and Q2)
        # Input dimension is embed_dim (for query) + embed_dim (for option) = embed_dim * 2
        # Output is a single Q-value per option for each Q-network.
        self.Q1_mlp = MLP(self.embed_dim * 2, q_hidden_dim, 1, layer_norm=True)
        self.Q2_mlp = MLP(self.embed_dim * 2, q_hidden_dim, 1, layer_norm=True)

    def forward_q_values(self,
                         emb_meta_task_mgr: torch.Tensor,          # (B, D_emb_meta_manager)
                         emb_global_context_mgr: torch.Tensor,     # (B, D_global)
                         obs_all_options_set_padded: torch.Tensor, # (B, max_total_options, D_option_feat)
                         all_options_padding_mask: Optional[torch.Tensor] = None # (B, max_total_options), boolean
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates Q1 and Q2 values for all options.

        Returns:
            q1_all_options (torch.Tensor): (B, max_total_options) Q1-values for each option.
            q2_all_options (torch.Tensor): (B, max_total_options) Q2-values for each option.
        """
        B = emb_meta_task_mgr.shape[0]

        # Step 1: Get contextual embeddings for all options using the shared extractor
        # seq_emb_options_contextual: (B, max_total_options, self.embed_dim)
        seq_emb_options_contextual = self.shared_extractor(obs_all_options_set_padded, all_options_padding_mask)

        # Step 2: Compute the query vector for the critic
        # query_embedding: (B, self.embed_dim)
        query_input = torch.cat([emb_meta_task_mgr, emb_global_context_mgr], dim=1)
        query_embedding = self.query_mlp(query_input)

        # Step 3: Prepare query for interaction with each option
        # query_expanded: (B, 1, self.embed_dim)
        query_expanded = query_embedding.unsqueeze(1)
        # query_broadcast: (B, max_total_options, self.embed_dim)
        query_broadcast = query_expanded.expand(-1, self.max_total_options, -1)

        # Step 4: Fuse query with each contextual option embedding
        # fused_input: (B, max_total_options, self.embed_dim * 2)
        fused_input = torch.cat([query_broadcast, seq_emb_options_contextual], dim=-1)

        # Step 5: Calculate Q1 and Q2 values for all options
        # q1_all_options_raw, q2_all_options_raw: (B, max_total_options, 1)
        q1_all_options_raw = self.Q1_mlp(fused_input)
        q2_all_options_raw = self.Q2_mlp(fused_input)

        # Squeeze the last dimension
        # q1_all_options, q2_all_options: (B, max_total_options)
        q1_all_options = q1_all_options_raw.squeeze(-1)
        q2_all_options = q2_all_options_raw.squeeze(-1)
        
        # Note: Unlike actor logits, Q-values for padded actions are usually NOT explicitly
        # masked to -inf here. The masking happens when selecting actions or targets.
        # If an algorithm requires masked Q-values (e.g. for a softmax over Q for some exploration),
        # it can be done externally. However, for standard Q-learning updates, raw Q-values are used.

        return q1_all_options, q2_all_options

    def q_for_action(self,
                     emb_meta_task_mgr: torch.Tensor,
                     emb_global_context_mgr: torch.Tensor,
                     obs_all_options_set_padded: torch.Tensor,
                     action_idx: torch.Tensor, # LongTensor (B,) or (B, 1)
                     all_options_padding_mask: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates Q1 and Q2 values for the chosen action indices.
        """
        q1_all, q2_all = self.forward_q_values(
            emb_meta_task_mgr,
            emb_global_context_mgr,
            obs_all_options_set_padded,
            all_options_padding_mask
        )

        # Ensure action_idx is (B, 1) for gather
        if action_idx.ndim == 1:
            action_idx_expanded = action_idx.unsqueeze(-1)
        else:
            action_idx_expanded = action_idx
            
        q1_selected = q1_all.gather(1, action_idx_expanded).squeeze(-1) # (B,)
        q2_selected = q2_all.gather(1, action_idx_expanded).squeeze(-1) # (B,)

        return q1_selected, q2_selected

# Example Usage (Conceptual - dimensions need to be defined)
if __name__ == '__main__':
    # Define dimensions
    D_option_feat_val = 8 
    encoder_hidden_dim_val = 64
    embed_dim_val = 128      
    num_attn_heads_val = 4 
    
    D_emb_meta_manager_val = 32 
    D_global_val = 16           
    max_total_options_val = 10  
    
    actor_query_hidden_dim_val = 64
    actor_scorer_hidden_dim_val = 64
    
    critic_query_hidden_dim_val = 128 # Critic can have different MLP sizes
    critic_q_hidden_dim_val = 128

    # Instantiate shared extractor
    shared_extractor = SharedFeatureExtractor(
        D_option_feat=D_option_feat_val,
        encoder_hidden_dim=encoder_hidden_dim_val,
        embed_dim=embed_dim_val,
        num_attn_heads=num_attn_heads_val
    )

    # Instantiate ManagerActor
    manager_actor = ManagerActor(
        shared_feature_extractor=shared_extractor,
        D_emb_meta_manager=D_emb_meta_manager_val,
        D_global=D_global_val,
        max_total_options=max_total_options_val,
        query_hidden_dim=actor_query_hidden_dim_val,
        scorer_hidden_dim=actor_scorer_hidden_dim_val
    )

    # Instantiate ManagerCritic
    manager_critic = ManagerCritic(
        shared_feature_extractor=shared_extractor,
        D_emb_meta_manager=D_emb_meta_manager_val,
        D_global=D_global_val,
        max_total_options=max_total_options_val,
        query_hidden_dim=critic_query_hidden_dim_val,
        q_hidden_dim=critic_q_hidden_dim_val
    )

    # Create dummy batch data
    B = 4 
    dummy_emb_meta_task_mgr = torch.randn(B, D_emb_meta_manager_val)
    dummy_emb_global_context_mgr = torch.randn(B, D_global_val)
    dummy_obs_all_options_set_padded = torch.randn(B, max_total_options_val, D_option_feat_val)
    
    dummy_all_options_padding_mask = torch.ones(B, max_total_options_val, dtype=torch.bool)
    num_valid_options_per_sample = [7, 6, 8, 7] 
    for i in range(B):
        dummy_all_options_padding_mask[i, :num_valid_options_per_sample[i]] = False

    # --- Test Actor ---
    print("--- Actor Test ---")
    logits = manager_actor.forward(
        dummy_emb_meta_task_mgr,
        dummy_emb_global_context_mgr,
        dummy_obs_all_options_set_padded,
        dummy_all_options_padding_mask
    )
    print(f"Actor Logits shape: {logits.shape}") 
    actions, log_probs, entropy = manager_actor.sample_action(
        dummy_emb_meta_task_mgr,
        dummy_emb_global_context_mgr,
        dummy_obs_all_options_set_padded,
        dummy_all_options_padding_mask
    )
    print(f"Sampled Actions shape: {actions.shape}")

    # --- Test Critic ---
    print("\n--- Critic Test ---")
    # Test forward_q_values
    q1_all, q2_all = manager_critic.forward_q_values(
        dummy_emb_meta_task_mgr,
        dummy_emb_global_context_mgr,
        dummy_obs_all_options_set_padded,
        dummy_all_options_padding_mask
    )
    print(f"Critic Q1_all shape: {q1_all.shape}") # Expected: (B, max_total_options)
    print(f"Critic Q2_all shape: {q2_all.shape}") # Expected: (B, max_total_options)
    print("Critic Q1_all example (first batch item where mask is False):")
    print(q1_all[0, ~dummy_all_options_padding_mask[0]])


    # Test q_for_action (using actions sampled from actor)
    q1_selected, q2_selected = manager_critic.q_for_action(
        dummy_emb_meta_task_mgr,
        dummy_emb_global_context_mgr,
        dummy_obs_all_options_set_padded,
        actions, # Use actions from actor
        dummy_all_options_padding_mask
    )
    print(f"\nCritic Q1_selected shape: {q1_selected.shape}") # Expected: (B,)
    print(f"Critic Q2_selected shape: {q2_selected.shape}") # Expected: (B,)
    print("Critic Q1_selected example:")
    print(q1_selected)