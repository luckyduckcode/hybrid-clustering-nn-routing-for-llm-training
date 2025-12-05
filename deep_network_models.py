"""
Deep Network Models for 1.58-Bit LLM Training

Advanced neural network architectures replacing linear models:
- Multi-layer transformers
- Attention mechanisms
- Residual connections
- Layer normalization
- Deep feed-forward networks

This module provides realistic, production-ready architectures for LLM training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import math


@dataclass
class TransformerConfig:
    """Configuration for Transformer models."""
    # Architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    
    # Quantization
    quantization_bits: float = 1.58
    use_quantized_attention: bool = False
    use_quantized_feedforward: bool = False


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, config: TransformerConfig):
        """Initialize multi-head attention."""
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        assert config.hidden_size % config.num_attention_heads == 0, \
            f"hidden_size ({config.hidden_size}) must be divisible by num_attention_heads ({config.num_attention_heads})"
        
        # Linear projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Output projection
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose for multi-head attention scores."""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: (batch_size, seq_length) or (batch_size, seq_length, seq_length)
        
        Returns:
            (attention_output, attention_weights)
        """
        batch_size = hidden_states.size(0)
        
        # Project to Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, -1, self.all_head_size)
        
        # Output projection
        attention_output = self.dense(context_layer)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        
        return attention_output, attention_probs


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, config: TransformerConfig):
        """Initialize feed-forward network."""
        super().__init__()
        
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = F.gelu
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
        
        Returns:
            output: (batch_size, seq_length, hidden_size)
        """
        residual = hidden_states
        
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Residual connection + layer norm
        output = self.LayerNorm(hidden_states + residual)
        
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer (attention + feed-forward)."""
    
    def __init__(self, config: TransformerConfig):
        """Initialize transformer layer."""
        super().__init__()
        
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask
        
        Returns:
            (layer_output, attention_weights)
        """
        attention_output, attention_weights = self.attention(
            hidden_states,
            attention_mask
        )
        
        layer_output = self.feed_forward(attention_output)
        
        return layer_output, attention_weights


class TransformerEncoder(nn.Module):
    """Multi-layer transformer encoder."""
    
    def __init__(self, config: TransformerConfig):
        """Initialize transformer encoder."""
        super().__init__()
        
        self.config = config
        self.layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through all layers.
        
        Args:
            hidden_states: (batch_size, seq_length, hidden_size)
            attention_mask: Attention mask
        
        Returns:
            (final_hidden_states, all_attention_weights)
        """
        all_attention_weights = []
        
        for layer_module in self.layers:
            hidden_states, attention_weights = layer_module(
                hidden_states,
                attention_mask
            )
            all_attention_weights.append(attention_weights)
        
        return hidden_states, all_attention_weights


class DeepTransformerLLM(nn.Module):
    """
    Deep Transformer-based LLM.
    
    Unlike linear models, this implements:
    - Embedding layers
    - Positional encoding
    - Multi-layer transformer encoder
    - Language model head
    """
    
    def __init__(self, config: TransformerConfig, vocab_size: int = 50257):
        """
        Initialize deep transformer LLM.
        
        Args:
            config: TransformerConfig
            vocab_size: Vocabulary size
        """
        super().__init__()
        
        self.config = config
        self.vocab_size = vocab_size
        
        # Embeddings
        self.word_embeddings = nn.Embedding(vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,
            config.hidden_size
        )
        
        self.embedding_LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(config)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_size, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(
                    mean=0.0,
                    std=self.config.initializer_range
                )
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(
                    mean=0.0,
                    std=self.config.initializer_range
                )
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_length) token indices
            attention_mask: (batch_size, seq_length) attention mask
            token_type_ids: (batch_size, seq_length) token type ids
        
        Returns:
            Dictionary with:
            - logits: (batch_size, seq_length, vocab_size)
            - hidden_states: (batch_size, seq_length, hidden_size)
            - attention_weights: List of attention weights from each layer
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Token type IDs (default to 0)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (batch_size, seq_length),
                dtype=torch.long,
                device=device
            )
        
        # Position IDs
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_LayerNorm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Transformer encoder
        hidden_states, attention_weights = self.encoder(
            embeddings,
            attention_mask
        )
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'attention_weights': attention_weights
        }


class DeepRNNLM(nn.Module):
    """
    Deep RNN-based Language Model.
    
    Alternative to transformer, uses:
    - Embedding layer
    - Multi-layer LSTM/GRU
    - Language model head
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        embedding_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 12,
        dropout: float = 0.1,
        rnn_type: str = 'lstm'
    ):
        """
        Initialize deep RNN LM.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
            rnn_type: 'lstm' or 'gru'
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # RNN layer
        rnn_class = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.output_dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        self.embedding.weight.data.normal_(mean=0, std=0.02)
        self.lm_head.weight.data.normal_(mean=0, std=0.02)
        self.lm_head.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_state: Optional[Tuple] = None
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_length)
            hidden_state: Previous hidden state
        
        Returns:
            Dictionary with:
            - logits: (batch_size, seq_length, vocab_size)
            - hidden_state: New hidden state
            - last_hidden_state: (batch_size, hidden_dim)
        """
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)
        
        # RNN
        output, hidden_state = self.rnn(embedded, hidden_state)
        
        # Output layer
        output = self.output_dropout(output)
        logits = self.lm_head(output)
        
        return {
            'logits': logits,
            'hidden_state': hidden_state,
            'last_hidden_state': output[:, -1, :].contiguous()
        }


# Model factory
def create_model(
    model_type: str = 'transformer',
    **kwargs
) -> nn.Module:
    """
    Create a deep LLM model.
    
    Args:
        model_type: 'transformer' or 'rnn'
        **kwargs: Model-specific arguments
    
    Returns:
        Model instance
    """
    if model_type == 'transformer':
        config = TransformerConfig(**{k: v for k, v in kwargs.items() if k in TransformerConfig.__dataclass_fields__})
        vocab_size = kwargs.get('vocab_size', 50257)
        return DeepTransformerLLM(config, vocab_size)
    
    elif model_type == 'rnn':
        return DeepRNNLM(
            vocab_size=kwargs.get('vocab_size', 50257),
            embedding_dim=kwargs.get('embedding_dim', 768),
            hidden_dim=kwargs.get('hidden_dim', 768),
            num_layers=kwargs.get('num_layers', 12),
            dropout=kwargs.get('dropout', 0.1),
            rnn_type=kwargs.get('rnn_type', 'lstm')
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
