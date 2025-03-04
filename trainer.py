import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

class MetricTracker:
    def __init__(self, metrics=None):
        self.metrics = metrics or ['loss', 'accuracy', 'precision', 'recall', 'f1']
        self.reset()
        
    def reset(self):
        self.train_metrics = {metric: [] for metric in self.metrics}
        self.val_metrics = {metric: [] for metric in self.metrics}
        self.epoch_train_metrics = {metric: 0 for metric in self.metrics}
        self.epoch_val_metrics = {metric: 0 for metric in self.metrics}
        self.best_val_metrics = {metric: 0 for metric in self.metrics}
        self.best_epoch = 0
        
    def update_train(self, metric, value):
        self.train_metrics[metric].append(value)
        
    def update_val(self, metric, value):
        self.val_metrics[metric].append(value)
        
    def epoch_average(self, phase='train'):
        metrics_dict = self.train_metrics if phase == 'train' else self.val_metrics
        epoch_metrics = self.epoch_train_metrics if phase == 'train' else self.epoch_val_metrics
        
        for metric in self.metrics:
            if metrics_dict[metric]:
                epoch_metrics[metric] = np.mean(metrics_dict[metric])
            
        if phase == 'val':
            if epoch_metrics['accuracy'] > self.best_val_metrics['accuracy']:
                self.best_val_metrics = epoch_metrics.copy()
                self.best_epoch = len(self.train_metrics['loss']) // len(self.train_metrics['loss'][0]) if isinstance(self.train_metrics['loss'][0], list) else len(self.train_metrics['loss'])
                return True
        return False
    
    def clear_batch_metrics(self):
        for metric in self.metrics:
            self.train_metrics[metric] = []
            self.val_metrics[metric] = []
            
    def get_epoch_metrics(self, phase='train'):
        return self.epoch_train_metrics if phase == 'train' else self.epoch_val_metrics
    
    def get_best_metrics(self):
        return self.best_val_metrics, self.best_epoch
    
    def plot_metrics(self, save_dir='plots'):
        os.makedirs(save_dir, exist_ok=True)
        
        epochs = range(1, len(self.epoch_train_metrics['loss']) + 1)
        
        for metric in self.metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, [self.epoch_train_metrics[metric][i] for i in range(len(epochs))], 'b-', label=f'Training {metric}')
            plt.plot(epochs, [self.epoch_val_metrics[metric][i] for i in range(len(epochs))], 'r-', label=f'Validation {metric}')
            plt.title(f'{metric.capitalize()} vs. Epochs')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'{metric}_plot.png'))
            plt.close()

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler=None, device=None, 
                 save_dir='checkpoints', log_dir='logs'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.save_dir = save_dir
        self.log_dir = log_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.metric_tracker = MetricTracker()
        self.log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
        
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metrics': self.metric_tracker.get_best_metrics()
        }
        
        torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_model.pth'))
            
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch']
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc="Training", total=len(train_loader)):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch.y)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
            self.metric_tracker.update_train('loss', loss.item())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        self.metric_tracker.update_train('accuracy', accuracy)
        self.metric_tracker.update_train('precision', precision)
        self.metric_tracker.update_train('recall', recall)
        self.metric_tracker.update_train('f1', f1)
        
        self.metric_tracker.epoch_average(phase='train')
        metrics = self.metric_tracker.get_epoch_metrics(phase='train')
        
        return metrics
    
    def validate(self, val_loader):
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", total=len(val_loader)):
                batch = batch.to(self.device)
                
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch.y)
                
                epoch_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                labels = batch.y.detach().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
                self.metric_tracker.update_val('loss', loss.item())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        self.metric_tracker.update_val('accuracy', accuracy)
        self.metric_tracker.update_val('precision', precision)
        self.metric_tracker.update_val('recall', recall)
        self.metric_tracker.update_val('f1', f1)
        
        is_best = self.metric_tracker.epoch_average(phase='val')
        metrics = self.metric_tracker.get_epoch_metrics(phase='val')
        
        return metrics, is_best
    
    def train(self, train_loader, val_loader, num_epochs=100, early_stopping_patience=10):
        self.log(f"Starting training on device: {self.device}")
        self.log(f"Model architecture:\n{self.model}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics, is_best = self.validate(val_loader)
            
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            log_message = f"Epoch {epoch}/{num_epochs} - Time: {epoch_time:.2f}s - "
            log_message += f"Train Loss: {train_metrics['loss']:.4f} - Train Acc: {train_metrics['accuracy']:.4f} - "
            log_message += f"Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.4f}"
            
            self.log(log_message)
            
            self.save_checkpoint(epoch, is_best)
            self.metric_tracker.clear_batch_metrics()
            
            if is_best:
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                self.log(f"Early stopping triggered after {epoch} epochs")
                break
        
        best_metrics, best_epoch = self.metric_tracker.get_best_metrics()
        self.log(f"Training completed. Best model at epoch {best_epoch} with metrics: {best_metrics}")
        self.metric_tracker.plot_metrics()
        
        return best_metrics, best_epoch
    
    def evaluate(self, test_loader, idx_to_class=None):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                outputs = self.model(batch)
                
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                labels = batch.y.detach().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        self.log(f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(12, 10))
        if idx_to_class:
            labels = [idx_to_class[i] for i in range(len(idx_to_class))]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        } 