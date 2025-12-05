"""
BERT Fine-tuning with 1.58-Bit Quantization (PyTorch)

Complete end-to-end example showing how to fine-tune BERT with
quantization-aware training using pytorch_integration.py

Requirements:
    pip install transformers torch

Usage:
    python bert_finetune_example.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from pytorch_integration import (
    QuantConfig,
    HybridTransformerWrapper,
    QuantizationAwareTrainer
)


class BERTFineTuneTask:
    """Complete BERT fine-tuning with quantization."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 2,
        max_seq_length: int = 128,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize BERT fine-tuning task."""
        self.device = device
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model for sequence classification
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        
        # Wrap with quantization
        self.quantization_config = QuantConfig(
            target_bits=1.58,
            adaptive_bits=True,
            min_bits=1.0,
            max_bits=8.0,
            enable_mixed_precision=False
        )
        
        self.model_wrapper = HybridTransformerWrapper(
            base_model,
            self.quantization_config
        )
        
        self.model = self.model_wrapper.model.to(device)
        
        print(f"Model loaded: {model_name}")
        print(f"Number of quantized layers: {self.model_wrapper.num_replaced_layers}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(
        self,
        texts: list,
        labels: list,
        batch_size: int = 32
    ) -> DataLoader:
        """
        Prepare data for training.
        
        Args:
            texts: List of text strings
            labels: List of class labels (0-indexed)
            batch_size: Batch size for training
        
        Returns:
            DataLoader with tokenized data
        """
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create dataset
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels, dtype=torch.long)
        )
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01
    ) -> dict:
        """
        Train model with quantization.
        
        Args:
            train_dataloader: Training data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            warmup_steps: Linear warmup steps
            weight_decay: Weight decay for regularization
        
        Returns:
            Training metrics dictionary
        """
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Get QAT trainer
        qat_trainer = self.model_wrapper.get_qat_trainer(optimizer)
        
        print("\n" + "="*60)
        print("Starting Quantization-Aware Fine-tuning")
        print("="*60 + "\n")
        
        total_steps = 0
        all_metrics = {
            'epoch_losses': [],
            'step_losses': [],
            'avg_bits_per_epoch': []
        }
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_quant_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                task_loss = outputs.loss
                
                # Compute quantization loss
                quant_loss = qat_trainer._compute_quantization_loss()
                total_loss = task_loss + 0.01 * quant_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Update statistics
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        qat_trainer.allocator.update_statistics(name, param.grad, param)
                
                epoch_losses.append(float(task_loss))
                epoch_quant_losses.append(float(quant_loss))
                total_steps += 1
                
                # Periodic bit allocation update
                if total_steps % 100 == 0:
                    qat_trainer.allocator.allocate_bits_adaptive()
                
                if (batch_idx + 1) % max(1, len(train_dataloader) // 5) == 0:
                    avg_bits = np.mean(
                        list(qat_trainer.allocator.bit_widths.values())
                    )
                    print(
                        f"Epoch {epoch+1}/{num_epochs}, "
                        f"Batch {batch_idx+1}/{len(train_dataloader)}, "
                        f"Loss: {total_loss:.4f}, "
                        f"Avg Bits: {avg_bits:.2f}"
                    )
            
            # Epoch summary
            epoch_avg_loss = np.mean(epoch_losses)
            epoch_avg_bits = np.mean(list(qat_trainer.allocator.bit_widths.values()))
            
            all_metrics['epoch_losses'].append(epoch_avg_loss)
            all_metrics['avg_bits_per_epoch'].append(epoch_avg_bits)
            
            print(
                f"\nEpoch {epoch+1} Summary:\n"
                f"  Average Loss: {epoch_avg_loss:.6f}\n"
                f"  Average Bits/Param: {epoch_avg_bits:.2f}\n"
            )
        
        print("="*60)
        print("Fine-tuning Complete")
        print("="*60 + "\n")
        
        return all_metrics
    
    def evaluate(self, eval_dataloader: DataLoader) -> dict:
        """
        Evaluate model on validation data.
        
        Args:
            eval_dataloader: Validation data
        
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += float(loss)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = correct / total
        
        self.model.train()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def save_model(self, path: str):
        """Save fine-tuned model."""
        self.model.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def get_model_size_info(self) -> dict:
        """Get model compression information."""
        return self.model_wrapper.get_model_bit_compression_ratio()


def create_synthetic_dataset(num_samples: int = 100) -> tuple:
    """
    Create synthetic sentiment analysis dataset for demo.
    
    Returns:
        (texts, labels) tuple
    """
    positive_texts = [
        "This movie is amazing and I loved it!",
        "Great product, highly recommended!",
        "The best experience I've had.",
        "Absolutely fantastic, I'm very happy.",
        "Wonderful service and great quality!"
    ]
    
    negative_texts = [
        "This is terrible and I hate it.",
        "Awful product, not recommended.",
        "The worst experience ever.",
        "Absolutely horrible, very disappointed.",
        "Poor quality and bad service."
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
    """Run BERT fine-tuning example."""
    print("BERT Fine-tuning with 1.58-Bit Quantization")
    print("=" * 60)
    
    # Initialize task
    task = BERTFineTuneTask(
        model_name='bert-base-uncased',
        num_classes=2,
        max_seq_length=128
    )
    
    # Create synthetic dataset (replace with real data)
    print("\nPreparing dataset...")
    train_texts, train_labels = create_synthetic_dataset(100)
    eval_texts, eval_labels = create_synthetic_dataset(50)
    
    train_dataloader = task.prepare_data(
        train_texts,
        train_labels,
        batch_size=16
    )
    
    eval_dataloader = task.prepare_data(
        eval_texts,
        eval_labels,
        batch_size=16
    )
    
    # Fine-tune with quantization
    metrics = task.train(
        train_dataloader,
        num_epochs=2,
        learning_rate=2e-5,
        weight_decay=0.01
    )
    
    # Evaluate
    print("Evaluating on validation set...")
    eval_metrics = task.evaluate(eval_dataloader)
    print(f"Validation Loss: {eval_metrics['loss']:.6f}")
    print(f"Validation Accuracy: {eval_metrics['accuracy']:.4f}")
    
    # Get compression info
    compression_info = task.get_model_size_info()
    print(f"\nModel Compression:")
    print(f"  Compression Ratio: {compression_info['compression_ratio']:.2f}x")
    print(f"  Average Bits: {compression_info['average_bits']:.2f}")
    print(f"  Original Size: {compression_info['original_size_mb']:.2f} MB")
    print(f"  Quantized Size: {compression_info['quantized_size_mb']:.2f} MB")
    
    # Save model
    task.save_model('./bert_quantized')
    
    print("\n" + "="*60)
    print("Fine-tuning Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
