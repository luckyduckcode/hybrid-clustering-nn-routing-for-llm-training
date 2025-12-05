"""
BERT Fine-tuning with 1.58-Bit Quantization (TensorFlow)

Complete end-to-end example showing how to fine-tune BERT with
quantization-aware training using tensorflow_integration.py

Requirements:
    pip install transformers tensorflow

Usage:
    python bert_tf_finetune_example.py
"""

import tensorflow as tf
import numpy as np
from typing import Tuple, List, Dict, Any
from tensorflow_integration import (
    TFQuantConfig,
    HuggingFaceTransformerIntegration,
    QuantizationAwareTrainerTF,
    compute_model_size_reduction_tf
)

try:
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
except ImportError:
    print("Note: Install transformers with: pip install transformers")


class BERTFineTuneTaskTF:
    """Complete BERT fine-tuning with TensorFlow quantization."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 2,
        max_seq_length: int = 128
    ):
        """Initialize BERT fine-tuning task for TensorFlow."""
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load TensorFlow model
        base_model = TFAutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        
        # Wrap with quantization
        self.quantization_config = TFQuantConfig(
            target_bits=1.58,
            adaptive_bits=True,
            min_bits=1.0,
            max_bits=8.0,
            quantize_weights=True,
            quantize_gradients=True,
            enable_mixed_precision=False
        )
        
        self.integration = HuggingFaceTransformerIntegration(
            base_model,
            self.quantization_config
        )
        
        self.model = self.integration.model
        
        print(f"Model loaded: {model_name}")
        print(f"Total parameters: {self.model.count_params():,}")
    
    def prepare_data(
        self,
        texts: List[str],
        labels: List[int],
        batch_size: int = 32,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Prepare data for training.
        
        Args:
            texts: List of text strings
            labels: List of class labels (0-indexed)
            batch_size: Batch size
            shuffle: Whether to shuffle data
        
        Returns:
            tf.data.Dataset
        """
        # Tokenize
        encodings = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask']
            },
            labels
        ))
        
        if shuffle:
            dataset = dataset.shuffle(len(texts))
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01
    ) -> Dict[str, Any]:
        """
        Train model with quantization awareness.
        
        Args:
            train_dataset: Training dataset
            num_epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization
        
        Returns:
            Training metrics
        """
        # Create optimizer with weight decay
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create QAT trainer
        qat_trainer = self.integration.get_qat_trainer(optimizer)
        
        # Loss function
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='mean'
        )
        
        print("\n" + "="*60)
        print("Starting Quantization-Aware Fine-tuning (TensorFlow)")
        print("="*60 + "\n")
        
        metrics_history = {
            'epoch_losses': [],
            'avg_bits_per_epoch': [],
            'all_losses': []
        }
        
        total_steps = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            epoch_losses = []
            batch_count = 0
            
            for batch_inputs, batch_targets in train_dataset:
                # Single training step
                step_metrics = qat_trainer.train_step(
                    batch_inputs['input_ids'],
                    batch_targets,
                    criterion
                )
                
                epoch_losses.append(step_metrics['total_loss'])
                metrics_history['all_losses'].append(step_metrics['total_loss'])
                
                batch_count += 1
                total_steps += 1
                
                # Print progress
                if batch_count % max(1, 5) == 0:
                    print(
                        f"  Batch {batch_count}: "
                        f"Loss={step_metrics['total_loss']:.4f}, "
                        f"AvgBits={step_metrics['avg_bits']:.2f}"
                    )
            
            # Epoch summary
            epoch_avg_loss = np.mean(epoch_losses)
            avg_bits = np.mean(
                list(qat_trainer.allocator.bit_widths.values())
            )
            
            metrics_history['epoch_losses'].append(epoch_avg_loss)
            metrics_history['avg_bits_per_epoch'].append(avg_bits)
            
            print(
                f"\nEpoch {epoch+1} Summary:\n"
                f"  Avg Loss: {epoch_avg_loss:.6f}\n"
                f"  Avg Bits/Param: {avg_bits:.2f}\n"
            )
        
        print("\n" + "="*60)
        print("Fine-tuning Complete")
        print("="*60 + "\n")
        
        return metrics_history
    
    def evaluate(self, eval_dataset: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate model on validation data.
        
        Args:
            eval_dataset: Validation dataset
        
        Returns:
            Evaluation metrics
        """
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_inputs, batch_targets in eval_dataset:
            outputs = self.model(
                input_ids=batch_inputs['input_ids'],
                attention_mask=batch_inputs['attention_mask'],
                training=False
            )
            
            logits = outputs.logits
            predictions = tf.argmax(logits, axis=1)
            
            # Compute loss
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                batch_targets,
                logits,
                from_logits=True
            )
            
            total_loss += tf.reduce_mean(loss).numpy()
            correct += tf.reduce_sum(
                tf.cast(predictions == batch_targets, tf.int32)
            ).numpy()
            total += batch_targets.shape[0]
        
        avg_loss = total_loss / len(eval_dataset)
        accuracy = correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def save_model(self, path: str):
        """Save fine-tuned model."""
        self.model.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def get_model_compression_info(self) -> Dict[str, Any]:
        """Get model compression statistics."""
        avg_bits = np.mean(
            list(self.integration.allocator.bit_widths.values())
        )
        return compute_model_size_reduction_tf(
            self.model,
            average_bits=avg_bits
        )


def create_synthetic_dataset(num_samples: int = 100) -> Tuple[List[str], List[int]]:
    """
    Create synthetic sentiment analysis dataset.
    
    Returns:
        (texts, labels) tuple
    """
    positive_texts = [
        "This is amazing and wonderful!",
        "Excellent product, highly recommended!",
        "Best experience ever!",
        "Absolutely fantastic!",
        "Great quality and service!"
    ]
    
    negative_texts = [
        "This is terrible and awful.",
        "Poor product, not recommended.",
        "Worst experience ever.",
        "Absolutely horrible.",
        "Bad quality and service."
    ]
    
    texts = []
    labels = []
    
    for i in range(num_samples):
        if i % 2 == 0:
            texts.append(positive_texts[i % len(positive_texts)])
            labels.append(1)
        else:
            texts.append(negative_texts[i % len(negative_texts)])
            labels.append(0)
    
    return texts, labels


def main():
    """Run BERT fine-tuning example with TensorFlow."""
    print("BERT Fine-tuning with 1.58-Bit Quantization (TensorFlow)")
    print("=" * 60)
    
    # Initialize task
    task = BERTFineTuneTaskTF(
        model_name='bert-base-uncased',
        num_classes=2,
        max_seq_length=128
    )
    
    # Prepare data
    print("\nPreparing dataset...")
    train_texts, train_labels = create_synthetic_dataset(100)
    eval_texts, eval_labels = create_synthetic_dataset(50)
    
    train_dataset = task.prepare_data(
        train_texts,
        train_labels,
        batch_size=16
    )
    
    eval_dataset = task.prepare_data(
        eval_texts,
        eval_labels,
        batch_size=16,
        shuffle=False
    )
    
    # Fine-tune
    metrics = task.train(
        train_dataset,
        num_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01
    )
    
    # Evaluate
    print("Evaluating on validation set...")
    eval_metrics = task.evaluate(eval_dataset)
    print(f"Validation Loss: {eval_metrics['loss']:.6f}")
    print(f"Validation Accuracy: {eval_metrics['accuracy']:.4f}")
    
    # Get compression info
    compression_info = task.get_model_compression_info()
    print(f"\nModel Compression:")
    print(f"  Total Parameters: {compression_info['total_params']:,}")
    print(f"  Original Size: {compression_info['original_size_mb']:.2f} MB")
    print(f"  Quantized Size: {compression_info['quantized_size_mb']:.2f} MB")
    print(f"  Reduction Ratio: {compression_info['reduction_ratio']:.2f}x")
    
    # Save model
    task.save_model('./bert_quantized_tf')
    
    print("\n" + "="*60)
    print("Fine-tuning Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
