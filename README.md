# RNA Sequence Classification with Graph Isomorphism Networks (GIN)

This project implements a Graph Isomorphism Network (GIN) for RNA sequence classification. The model converts RNA sequences into graph structures and uses graph neural networks to classify them into different RNA classes.

## Features

- Conversion of RNA sequences to graph representations
- Graph Isomorphism Network (GIN) implementation for RNA classification
- Comprehensive training pipeline with metrics tracking
- Real-time logging of training progress
- Visualization of training metrics and confusion matrices
- Support for early stopping and learning rate scheduling

## Project Structure

- `data_processing.py`: Functions for data loading, preprocessing, and graph conversion
- `model.py`: GIN model architecture implementation
- `trainer.py`: Training and evaluation pipeline with MetricTracker
- `main.py`: Main script to run the training process
- `requirements.txt`: Required dependencies

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rna-gin-classification.git
cd rna-gin-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script with default parameters:
```bash
python main.py
```

Or customize the training with command-line arguments:
```bash
python main.py --data_path data/ECCB2017/dataset_Rfam_validated_2600_13classes.fasta --hidden_dim 256 --batch_size 64 --epochs 200
```

### Command-line Arguments

- `--data_path`: Path to the FASTA file (default: "data/ECCB2017/dataset_Rfam_validated_2600_13classes.fasta")
- `--test_size`: Proportion of data to use for testing (default: 0.2)
- `--val_size`: Proportion of training data to use for validation (default: 0.1)
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--num_layers`: Number of GIN layers (default: 3)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for regularization (default: 1e-5)
- `--patience`: Patience for early stopping (default: 10)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--seed`: Random seed for reproducibility (default: 42)
- `--save_dir`: Directory to save model checkpoints (default: "checkpoints")
- `--log_dir`: Directory to save training logs (default: "logs")

## Data Format

The model expects RNA sequences in FASTA format with headers containing class information:
```
>RF00001_AF095839_1_346-228 5S_rRNA
GCGTACGGCCATACTATGGGGAATACACCTGATCCCGTCCGATTTCAGAAGTTAAGCCTC
ATCAGGCATCCTAAGTACTAGGGTGGGCGACCACCTGGGAACCGGATGTGCTGTACGCT
```

## Model Architecture

The model converts RNA sequences into graphs where:
- Nodes represent k-mers from the sequence
- Edges connect k-mers that are close in the sequence
- Node features are one-hot encodings of nucleotides

The GIN architecture then processes these graphs to learn meaningful representations for classification.

## Results

Training results, including metrics and visualizations, are saved in the specified directories:
- Model checkpoints: `checkpoints/`
- Training logs: `logs/`
- Metric plots: `plots/`

## License

MIT 