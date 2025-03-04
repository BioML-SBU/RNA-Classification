import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os

from data_processing import load_data, create_dataloader
from model import RNAClassifier
from trainer import Trainer

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load and preprocess data
    train_dataset, test_dataset = load_data(args.train_data_path, args.test_data_path)  # Load both datasets
    
    # Create dataloaders
    train_loader, test_loader = create_dataloader(train_dataset, test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Calculate input dimension based on k-mer size
    k = 4  # k-mer size used in sequence_to_graph
    input_dim = 5 * k  # 5 is the one-hot encoding dimension for nucleotides
    
    # Create model
    model = RNAClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=train_dataset.num_classes,
        num_layers=args.num_layers
    )
    
    # Define loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    # Train model
    best_metrics, best_epoch = trainer.train(
        train_loader=train_loader,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience
    )
    
    # Load best model and evaluate on test set
    trainer.load_checkpoint(os.path.join(args.save_dir, 'best_model.pth'))
    test_metrics = trainer.evaluate(test_loader, idx_to_class=test_dataset.idx_to_class)
    
    print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNA Classification with GIN")
    parser.add_argument("--train_data_path", type=str, default="data/train_data.fasta",
                        help="Path to the training FASTA file")
    parser.add_argument("--test_data_path", type=str, default="data/test_data.fasta",
                        help="Path to the test FASTA file")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of GIN layers")
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for regularization")
    parser.add_argument("--patience", type=int, default=10,
                        help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save training logs")
    
    args = parser.parse_args()
    
    main(args)
