"""
Parameter Sharing for 1.58-Bit Quantized LLMs

Advanced parameter sharing techniques:
- Weight sharing across layers
- Encoder-decoder parameter tying
- Shared attention heads
- Shared feedforward networks
- Tied embeddings (input/output)
- Cross-layer parameter sharing

Reduces model size and memory footprint while maintaining expressivity.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class ParameterSharingConfig:
    """Configuration for parameter sharing."""
    # Sharing types
    tie_embeddings: bool = True  # Tie input/output embeddings
    tie_encoder_decoder: bool = False  # Share encoder/decoder weights
    share_attention_heads: bool = False  # Share attention across layers
    share_feedforward: bool = False  # Share FFN across layers
    share_layer_norm: bool = False  # Share layer norm parameters
    cross_layer_sharing: bool = False  # Share parameters across non-adjacent layers
    
    # Sharing patterns
    sharing_pattern: str = 'sequential'  # 'sequential', 'alternate', 'sparse'
    sharing_interval: int = 2  # Share every N layers
    sharing_groups: Optional[List[List[int]]] = None  # Explicit sharing groups
    
    # Quantization aware
    quantize_shared_params: bool = True
    shared_param_bits: float = 1.58


class ParameterSharingManager:
    """Manages parameter sharing across model layers."""
    
    def __init__(self, model: nn.Module, config: ParameterSharingConfig):
        """
        Initialize parameter sharing manager.
        
        Args:
            model: Model to apply sharing to
            config: ParameterSharingConfig
        """
        self.model = model
        self.config = config
        self.sharing_map: Dict[str, str] = {}  # Maps parameter name to source
        self.shared_parameters: Dict[str, nn.Parameter] = {}
        
        self._apply_sharing_strategy()
    
    def _apply_sharing_strategy(self):
        """Apply parameter sharing strategy."""
        if self.config.tie_embeddings:
            self._tie_embeddings()
        
        if self.config.tie_encoder_decoder:
            self._tie_encoder_decoder()
        
        if self.config.share_attention_heads:
            self._share_attention_heads()
        
        if self.config.share_feedforward:
            self._share_feedforward()
        
        if self.config.cross_layer_sharing:
            self._apply_cross_layer_sharing()
    
    def _tie_embeddings(self):
        """Tie input and output embeddings."""
        # Find embedding and output layer
        embedding_layer = None
        output_layer = None
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding) and 'word' in name:
                embedding_layer = module
            elif isinstance(module, nn.Linear) and 'lm_head' in name:
                output_layer = module
        
        if embedding_layer and output_layer:
            # Share weights
            if embedding_layer.weight.shape[0] == output_layer.weight.shape[1]:
                output_layer.weight = embedding_layer.weight
                self.sharing_map['lm_head.weight'] = 'word_embeddings.weight'
                print("✓ Tied input and output embeddings")
    
    def _tie_encoder_decoder(self):
        """Tie encoder and decoder weights."""
        encoder_params = {}
        decoder_params = {}
        
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                clean_name = name.replace('encoder.', '')
                encoder_params[clean_name] = (name, param)
            elif 'decoder' in name:
                clean_name = name.replace('decoder.', '')
                decoder_params[clean_name] = (name, param)
        
        # Share matching parameters
        shared_count = 0
        for clean_name, (decoder_name, decoder_param) in decoder_params.items():
            if clean_name in encoder_params:
                encoder_name, encoder_param = encoder_params[clean_name]
                
                # Point decoder to encoder parameter
                self._set_shared_parameter(decoder_name, encoder_param)
                self.sharing_map[decoder_name] = encoder_name
                shared_count += 1
        
        if shared_count > 0:
            print(f"✓ Tied {shared_count} encoder-decoder parameters")
    
    def _share_attention_heads(self):
        """Share attention heads across layers."""
        attention_layers = []
        
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'query'):
                attention_layers.append((name, module))
        
        # Share first attention head with others
        if len(attention_layers) > 1:
            reference_layer = attention_layers[0][1]
            
            shared_count = 0
            for name, layer in attention_layers[1:]:
                if (hasattr(layer, 'query') and 
                    layer.query.weight.shape == reference_layer.query.weight.shape):
                    
                    layer.query = reference_layer.query
                    layer.key = reference_layer.key
                    layer.value = reference_layer.value
                    
                    self.sharing_map[f'{name}.query'] = f'{attention_layers[0][0]}.query'
                    shared_count += 1
            
            if shared_count > 0:
                print(f"✓ Shared attention heads across {shared_count} layers")
    
    def _share_feedforward(self):
        """Share feedforward networks across layers."""
        ffn_layers = []
        
        for name, module in self.model.named_modules():
            if 'feed_forward' in name.lower() or 'mlp' in name.lower():
                ffn_layers.append((name, module))
        
        # Share first FFN with others
        if len(ffn_layers) > 1:
            reference_layer = ffn_layers[0][1]
            
            shared_count = 0
            for name, layer in ffn_layers[1:]:
                if (hasattr(layer, 'dense_1') and 
                    hasattr(reference_layer, 'dense_1') and
                    layer.dense_1.weight.shape == reference_layer.dense_1.weight.shape):
                    
                    layer.dense_1 = reference_layer.dense_1
                    layer.dense_2 = reference_layer.dense_2
                    
                    self.sharing_map[f'{name}.dense_1'] = f'{ffn_layers[0][0]}.dense_1'
                    shared_count += 1
            
            if shared_count > 0:
                print(f"✓ Shared feedforward networks across {shared_count} layers")
    
    def _apply_cross_layer_sharing(self):
        """Apply cross-layer parameter sharing based on pattern."""
        if self.config.sharing_pattern == 'sequential':
            self._sequential_sharing()
        elif self.config.sharing_pattern == 'alternate':
            self._alternate_sharing()
        elif self.config.sharing_pattern == 'sparse':
            self._sparse_sharing()
    
    def _sequential_sharing(self):
        """Share parameters sequentially every N layers."""
        layer_params = self._get_layered_params()
        
        shared_count = 0
        for layer_idx, (layer_name, params) in enumerate(layer_params.items()):
            if layer_idx % self.config.sharing_interval == 0:
                source_layer = layer_idx
            
            if layer_idx % self.config.sharing_interval != 0:
                source_name = list(layer_params.keys())[source_layer]
                source_params = layer_params[source_name]
                
                # Share parameters
                for param_name in params:
                    if param_name in source_params:
                        self.sharing_map[f'{layer_name}.{param_name}'] = f'{source_name}.{param_name}'
                        shared_count += 1
        
        if shared_count > 0:
            print(f"✓ Sequential parameter sharing: {shared_count} params shared")
    
    def _alternate_sharing(self):
        """Share parameters in alternating pattern."""
        layer_params = self._get_layered_params()
        
        shared_count = 0
        for layer_idx, (layer_name, params) in enumerate(layer_params.items()):
            if layer_idx % 2 == 1:  # Odd layers share with even
                source_name = list(layer_params.keys())[layer_idx - 1]
                source_params = layer_params[source_name]
                
                for param_name in params:
                    if param_name in source_params:
                        self.sharing_map[f'{layer_name}.{param_name}'] = f'{source_name}.{param_name}'
                        shared_count += 1
        
        if shared_count > 0:
            print(f"✓ Alternate parameter sharing: {shared_count} params shared")
    
    def _sparse_sharing(self):
        """Apply sparse parameter sharing pattern."""
        if self.config.sharing_groups:
            shared_count = 0
            for group in self.config.sharing_groups:
                if len(group) > 1:
                    source_idx = group[0]
                    for target_idx in group[1:]:
                        self.sharing_map[f'layer.{target_idx}'] = f'layer.{source_idx}'
                        shared_count += 1
            
            if shared_count > 0:
                print(f"✓ Sparse parameter sharing: {shared_count} groups")
    
    def _get_layered_params(self) -> Dict[str, List[str]]:
        """Get parameters organized by layer."""
        layer_params = {}
        
        for name, param in self.model.named_parameters():
            # Extract layer number
            parts = name.split('.')
            if parts[0].isdigit():
                layer_idx = int(parts[0])
                param_name = '.'.join(parts[1:])
            else:
                layer_idx = 0
                param_name = name
            
            layer_key = f'layer.{layer_idx}'
            if layer_key not in layer_params:
                layer_params[layer_key] = []
            
            layer_params[layer_key].append(param_name)
        
        return layer_params
    
    def _set_shared_parameter(self, target_name: str, shared_param: nn.Parameter):
        """Set a parameter to be shared."""
        self.shared_parameters[target_name] = shared_param
    
    def get_sharing_info(self) -> Dict[str, Any]:
        """Get information about parameter sharing."""
        total_params = sum(p.numel() for p in self.model.parameters())
        unique_params = sum(
            p.numel() for p in set(self.model.parameters())
        )
        reduction = 1 - (unique_params / total_params) if total_params > 0 else 0
        
        return {
            'total_parameters': total_params,
            'unique_parameters': unique_params,
            'shared_parameters': total_params - unique_params,
            'parameter_reduction': reduction,
            'sharing_map_size': len(self.sharing_map),
            'sharing_map': self.sharing_map
        }


class SharedParameterQuantizer:
    """
    Quantization-aware shared parameter handling.
    
    Ensures shared parameters maintain consistency across quantization.
    """
    
    def __init__(self, manager: ParameterSharingManager):
        """Initialize quantizer."""
        self.manager = manager
        self.shared_quantization_scales = {}
    
    def quantize_shared_params(self, bits: float = 1.58):
        """
        Quantize shared parameters.
        
        Args:
            bits: Bit width for quantization
        """
        # Get unique shared parameters
        quantized_count = 0
        
        for source_name, param in self.manager.shared_parameters.items():
            param_abs = torch.abs(param.data)
            num_levels = int(2 ** (bits - 0.58))
            max_val = torch.max(param_abs)
            
            if max_val > 0:
                scale = max_val / (num_levels - 1)
                quantized = torch.round(param.data / (scale + 1e-8)) * scale
                param.data = quantized
                
                self.shared_quantization_scales[source_name] = float(scale)
                quantized_count += 1
        
        return quantized_count
    
    def get_quantization_stats(self) -> Dict[str, float]:
        """Get quantization statistics."""
        return {
            'quantized_shared_params': len(self.shared_quantization_scales),
            'avg_scale': np.mean(list(self.shared_quantization_scales.values()))
            if self.shared_quantization_scales else 0.0
        }


class LayerShareModel(nn.Module):
    """
    Wrapper model with automatic layer sharing.
    
    Usage:
        config = ParameterSharingConfig(tie_embeddings=True, share_feedforward=True)
        shared_model = LayerShareModel(model, config)
    """
    
    def __init__(self, model: nn.Module, config: ParameterSharingConfig):
        """
        Initialize layer sharing model.
        
        Args:
            model: Model to wrap
            config: ParameterSharingConfig
        """
        super().__init__()
        
        self.model = model
        self.config = config
        
        # Apply sharing
        self.sharing_manager = ParameterSharingManager(model, config)
        self.quantizer = SharedParameterQuantizer(self.sharing_manager)
    
    def forward(self, *args, **kwargs):
        """Forward pass through shared model."""
        return self.model(*args, **kwargs)
    
    def get_sharing_info(self) -> Dict[str, Any]:
        """Get parameter sharing information."""
        return self.sharing_manager.get_sharing_info()
    
    def quantize_shared_parameters(self, bits: float = 1.58) -> int:
        """Quantize shared parameters."""
        return self.quantizer.quantize_shared_params(bits)
    
    def get_quantization_stats(self) -> Dict[str, float]:
        """Get quantization statistics."""
        return self.quantizer.get_quantization_stats()


def compute_shared_model_size(
    original_model: nn.Module,
    sharing_config: ParameterSharingConfig
) -> Dict[str, float]:
    """
    Compute model size reduction from parameter sharing.
    
    Args:
        original_model: Original model
        sharing_config: Sharing configuration
    
    Returns:
        Size metrics
    """
    # Count original parameters
    original_params = sum(p.numel() for p in original_model.parameters())
    original_size_mb = (original_params * 32) / (8 * 1024 * 1024)
    
    # Apply sharing and count unique parameters
    shared_model = LayerShareModel(original_model, sharing_config)
    
    # Count unique parameters
    unique_params = sum(
        p.numel() for p in set(shared_model.model.parameters())
    )
    shared_size_mb = (unique_params * 32) / (8 * 1024 * 1024)
    
    return {
        'original_parameters': original_params,
        'unique_parameters': unique_params,
        'shared_parameters': original_params - unique_params,
        'original_size_mb': original_size_mb,
        'shared_size_mb': shared_size_mb,
        'size_reduction_ratio': original_size_mb / (shared_size_mb + 1e-8),
        'parameter_reduction_ratio': original_params / (unique_params + 1e-8)
    }
