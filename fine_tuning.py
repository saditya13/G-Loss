import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import torch.utils.data as Data
from ignite.engine import Events, Engine
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score,classification_report, brier_score_loss
import time
from datetime import datetime
import numpy as np
import os, sys
from datetime import datetime
import argparse, shutil, logging
from torch.optim import lr_scheduler
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import datetime
import time
now = datetime.datetime.now()
import losses
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--nb_epochs', type=int, default=80)
parser.add_argument('--bert_lr', type=float, default=None, help='learning rate (will use best from optuna if not specified)')
parser.add_argument('--dataset', default='ohsumed', choices=['20ng', 'R8', 'R52', 'ohsumed', 'MR','MR_toy', 'R8_toy','R52_toy','ohsumed_toy','20ng_toy'])
parser.add_argument('--bert_init', type=str, default='bert-base-uncased',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[dataset] if not specified')
parser.add_argument('--iter', type=int, default = 20)
parser.add_argument('--gamma', type=float, default=None, help='gamma parameter (will use best from optuna if not specified)')
parser.add_argument('--sigma', type=float, default=None, help='sigma parameter (will use best from optuna if not specified)')
parser.add_argument('--loss', type=str, default='scl', choices=['cross_entropy', 'g-loss', 'scl', 'triplet','cos-sim'])
parser.add_argument('--temperature', type=float, default=None, help='temperature for SCL loss (will use best from optuna if not specified)')
parser.add_argument('--optuna_results', type=str, default='optuna_best_params.json', help='JSON file with best parameters from Optuna')
parser.add_argument('--use_latest_checkpoint', action='store_true', help='Automatically use the most recent checkpoint for the model configuration')

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
nb_epochs = args.nb_epochs
data = args.dataset
bert_init = args.bert_init
checkpoint_dir = args.checkpoint_dir
iter = args.iter
loss_function = args.loss
use_latest_checkpoint = args.use_latest_checkpoint  # New argument


def find_latest_checkpoint(base_dir='./checkpoint', loss_function=None, bert_init=None, data=None):
    
    # Find the most recent checkpoint directory for a given model configuration.

    # Args:
    #     base_dir: Base directory where checkpoints are stored
    #     loss_function: Loss function name (e.g., 'scl', 'triplet')
    #     bert_init: BERT initialization model name (e.g., 'bert-base-uncased')
    #     data: Dataset name (e.g., 'ohsumed', '20ng')

    # Returns:
    #     Path to the most recent checkpoint directory, or None if no matching directory found
    
    import os
    import glob
    from datetime import datetime
    import logging

    # Create a simple print function instead of using logger
    def log_info(message):
        print(f"[INFO] {message}")

    # Create a pattern to match directories
    pattern_parts = []
    if loss_function:
        pattern_parts.append(loss_function)
    if bert_init:
        pattern_parts.append(bert_init)
    if data:
        pattern_parts.append(data)

    pattern = "_".join(pattern_parts)
    if pattern:
        pattern = f"{pattern}_*"
    else:
        pattern = "*"

    # Get all matching directories
    search_path = os.path.join(base_dir, pattern)
    matching_dirs = glob.glob(search_path)

    if not matching_dirs:
        log_info(f"No matching checkpoint directories found for pattern: {pattern}")
        return None

    # Extract timestamps and sort
    timestamp_format = "%Y-%m-%d-%H-%M-%S"

    def extract_timestamp(dir_path):
        # Extract the timestamp part from the directory name
        dir_name = os.path.basename(dir_path)
        timestamp_str = dir_name.split('_')[-1]
        try:
            return datetime.strptime(timestamp_str, timestamp_format)
        except ValueError:
            # If timestamp can't be parsed, return a very old date
            log_info(f"Warning: Could not parse timestamp from directory: {dir_name}")
            return datetime(1900, 1, 1)

    # Sort by timestamp (newest first)
    sorted_dirs = sorted(matching_dirs, key=extract_timestamp, reverse=True)

    latest_dir = sorted_dirs[0] if sorted_dirs else None
    if latest_dir:
        log_info(f"Found latest checkpoint directory: {latest_dir}")

    # Return the most recent directory
    return latest_dir


if checkpoint_dir is None and use_latest_checkpoint:
    # Try to find the latest checkpoint
    latest_ckpt = find_latest_checkpoint('./checkpoint', loss_function, bert_init, data)
    if latest_ckpt:
        ckpt_dir = latest_ckpt
        print(f"Using latest checkpoint directory: {ckpt_dir}")  # Print before logger is set up
    else:
        # Create a new checkpoint directory
        ckpt_dir = './checkpoint/{}_{}_{}_{}'.format(loss_function, bert_init, data, now.strftime("%Y-%m-%d-%H-%M-%S"))
        print(f"No existing checkpoint found. Creating new directory: {ckpt_dir}")  # Print before logger is set up
elif checkpoint_dir is None:
    # Original behavior when no checkpoint is specified and not using latest
    ckpt_dir = './checkpoint/{}_{}_{}_{}'.format(loss_function, bert_init, data, now.strftime("%Y-%m-%d-%H-%M-%S"))
else:
    ckpt_dir = checkpoint_dir

# After this point, your existing code for setting up the logger continues
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training_finetuning_plotting.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

# After the logger is set up, you can use it to log the checkpoint directory
logger.info('Arguments:')
logger.info(str(args))
logger.info(f"Using checkpoint directory: {ckpt_dir}")

# Load the best parameters from Optuna results if available
import json
try:
    # First look for Optuna results in the checkpoint directory
    optuna_results_path = os.path.join(ckpt_dir, "optuna_best_params.json")
    print("best parameter path:",optuna_results_path)
    if os.path.exists(optuna_results_path):
        logger.info(f"Found Optuna results in checkpoint directory: {optuna_results_path}")
    else:
        # Fall back to the specified path
        optuna_results_path = args.optuna_results
        logger.info(f"Looking for Optuna results at: {optuna_results_path}")

    with open(optuna_results_path, 'r') as f:
        best_params = json.load(f)
    logger.info(f"Loaded best parameters from {optuna_results_path}: {best_params}")

    # Set parameters based on the loss function, with command line args taking precedence
    if loss_function == 'scl':
        temperature = args.temperature if args.temperature is not None else best_params.get('temperature', 0.1)
        bert_lr = args.bert_lr if args.bert_lr is not None else best_params.get('learning_rate', 3e-5)
        gamma = args.gamma  # Not used for SCL but keeping for consistency
        sigma = args.sigma  # Not used for SCL but keeping for consistency
    elif loss_function == 'g-loss':
        gamma = args.gamma if args.gamma is not None else best_params.get('gamma', 0.8)
        sigma = args.sigma if args.sigma is not None else best_params.get('sigma', 0.5563)
        bert_lr = args.bert_lr if args.bert_lr is not None else best_params.get('learning_rate', 3e-5)
        temperature = args.temperature  # Not used for g-loss but keeping for consistency
    else:
        # For other loss functions, just use the learning rate if available
        bert_lr = args.bert_lr if args.bert_lr is not None else best_params.get('learning_rate', 3e-5)
        gamma = args.gamma if args.gamma is not None else 0.8
        sigma = args.sigma if args.sigma is not None else 0.5563
        temperature = args.temperature if args.temperature is not None else 0.1

except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.info(f"Could not load Optuna results: {str(e)}")
    logger.info(f"Using default or provided values")
    # Fall back to command line arguments or defaults
    bert_lr = args.bert_lr if args.bert_lr is not None else 3e-5
    gamma = args.gamma if args.gamma is not None else 0.8
    sigma = args.sigma if args.sigma is not None else 0.5563
    temperature = args.temperature if args.temperature is not None else 0.1

logger.info(f"Using parameters: learning_rate={bert_lr}, gamma={gamma}, sigma={sigma}, temperature={temperature}")

cpu = torch.device('cpu')
gpu = torch.device('cuda:0')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

###DATA###
data_dir = '/home/aditya/graph_based_loss/data/' + data +'/'
# data_dir = '/home/aditya/graph_based_loss/data/toy_datasets/' + data + '/'

input_ids, attention_mask, labels = {}, {}, {}

train = pd.read_csv(os.path.join(data_dir, 'train.csv')).sample(frac=1).reset_index(drop=True)
val = pd.read_csv(os.path.join(data_dir, 'val.csv')).sample(frac=1).reset_index(drop=True)
test = pd.read_csv(os.path.join(data_dir, 'test.csv')).sample(frac=1).reset_index(drop=True)
num_labels = len(np.unique(train['label']))  # Will automatically set based on dataset

tokenizer = AutoTokenizer.from_pretrained(bert_init)
model = AutoModel.from_pretrained(bert_init)

EPS = 1e-9

def encode_input(text, tokenizer):
    encoded = tokenizer(list(text), max_length=max_length, truncation=True, padding="max_length", return_tensors='pt')
    return encoded.input_ids, encoded.attention_mask

# Tokenize the text data
train_input_ids, train_attention_mask = encode_input(train["text"], tokenizer)
val_input_ids, val_attention_mask = encode_input(val["text"], tokenizer)
test_input_ids, test_attention_mask = encode_input(test["text"], tokenizer)

#Label Encoding
label_encoder = LabelEncoder()
train['label'] = label_encoder.fit_transform(train['label'])
val['label'] = label_encoder.fit_transform(val['label'])
test['label'] = label_encoder.fit_transform(test['label'])

# Convert labels to tensors
labels['train'] = torch.tensor(train["label"].values, dtype=torch.long)
labels['val'] = torch.tensor(val["label"].values, dtype=torch.long)
labels['test'] = torch.tensor(test["label"].values, dtype=torch.long)

# Store tokenized data
input_ids['train'], input_ids['val'], input_ids['test'] = train_input_ids, val_input_ids, test_input_ids
attention_mask['train'], attention_mask['val'], attention_mask['test'] = train_attention_mask, val_attention_mask, test_attention_mask

# Create DataLoaders
datasets, loader = {}, {}
for split in ['train', 'val', 'test']:
    datasets[split] = Data.TensorDataset(input_ids[split], attention_mask[split], labels[split])
    loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train['label']), y=train['label'])
print(class_weights)

def evaluate_clustering_unsupervised(embeddings):
    # Use fixed number of clusters equal to the number of classes
    n_clusters = num_labels  # Already defined globally based on dataset

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings)

    # Calculate silhouette score
    sil_score = silhouette_score(embeddings, predicted_labels) if len(np.unique(predicted_labels)) > 1 else 0

    metrics = {
        'silhouette': sil_score,
        'predicted_labels': predicted_labels,
        'n_clusters': n_clusters
    }

    return metrics

class UnsupervisedMetricsTracker:
    def __init__(self, model, ckpt_dir):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.silhouette_scores = []
        self.n_clusters_history = []
        self.best_embeddings = None
        self.best_labels = None
        self.best_epoch = -1
        self.best_silhouette = -1
        self.last_epoch_time = time.time()  # Initialize the timestamp

        # For early stopping
        self.patience = 10  # Number of epochs with no improvement                     ########## patience changed ########
        self.min_delta = 0.005  # Minimum change to qualify as improvement
        self.plateau_counter = 0

    def create_tsne_plot(self, epoch, embeddings, cluster_labels):
        
        # Create and save t-SNE plot for the embeddings with cluster labels
        # Args:
        #     epoch: current epoch number
        #     embeddings: the embeddings to visualize
        #     cluster_labels: cluster labels from unsupervised clustering
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        tsne_result = tsne.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            indices = cluster_labels == label
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                        c=[colors[i]], label=f'Cluster {label}', alpha=0.7)

        plt.title(f't-SNE Visualization - Epoch {epoch}')
        # plt.legend(loc='best')
        # plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.ckpt_dir, f'tsne_plot_epoch_{epoch}.png'))
        plt.close()

    def evaluate_epoch(self, epoch):

        self.model.eval()
        with torch.no_grad():
          # Get train embeddings
          train_embeddings = []

          for batch in loader['train']:
              input_ids_batch, attention_mask_batch, _ = [x.to(gpu) for x in batch]
              outputs = self.model(input_ids_batch, attention_mask_batch, output_hidden_states=True)
              embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Get CLS token
              train_embeddings.append(embeddings)

          train_embeddings = np.vstack(train_embeddings)

          # Normalize embeddings
          train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)

          # Evaluate clustering using unsupervised metrics
          metrics = evaluate_clustering_unsupervised(train_embeddings)

          # Store metrics
          self.silhouette_scores.append(metrics['silhouette'])
          self.n_clusters_history.append(metrics['n_clusters'])

          # Log metrics
          logger.info(
              f"Epoch {epoch} Unsupervised Metrics: "
              f"Silhouette={metrics['silhouette']:.4f}, "
              f"Optimal Clusters={metrics['n_clusters']}"
          )

          # Check if this is the best epoch based on silhouette score
          current_score = metrics['silhouette']
          if current_score > self.best_silhouette:
              self.best_silhouette = current_score
              self.best_embeddings = train_embeddings.copy()
              self.best_labels = metrics['predicted_labels']
              self.best_epoch = epoch
              self.plateau_counter = 0

              # Create t-SNE plot for the best embeddings
              self.create_tsne_plot(epoch, train_embeddings, metrics['predicted_labels'])

              return {
                  'is_best': True,
                  'early_stop': False
              }
          else:
              # Check for improvement within delta
              if self.best_silhouette - current_score <= self.min_delta:
                  # Small or no deterioration
                  self.plateau_counter += 1
              else:
                  # Significant deterioration, reset counter
                  self.plateau_counter = 0

              # Check for early stopping
              if self.plateau_counter >= self.patience:
                  logger.info(f"Early stopping triggered at epoch {epoch}: "
                            f"No improvement in silhouette score for {self.patience} epochs")
                  return {
                      'is_best': False,
                      'early_stop': True
                  }

              return {
                  'is_best': False,
                  'early_stop': False
              }

    def on_start(self):
        self.last_epoch_time = time.time()

    def plot_metrics(self):
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.silhouette_scores) + 1), self.silhouette_scores, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Score vs Epoch for {loss_function}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.ckpt_dir, 'silhouette_score_plot.png'))
        plt.close()

        # Plot optimal clusters
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.n_clusters_history) + 1), self.n_clusters_history, marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Clusters')
        plt.title(f'Optimal Number of Clusters vs Epoch for {loss_function}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.ckpt_dir, 'optimal_clusters_plot.png'))
        plt.close()

        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(self.silhouette_scores) + 1),
            'silhouette': self.silhouette_scores,
            'optimal_clusters': self.n_clusters_history
        })
        metrics_df.to_csv(os.path.join(self.ckpt_dir, 'unsupervised_metrics.csv'), index=False)


class FinalEvaluator:
    def __init__(self, model, ckpt_dir):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.hidden_size = model.config.hidden_size

    def _encode(self, input_ids, attention_mask, batch_size):
        input_ids, attention_mask = input_ids.to(gpu), attention_mask.to(gpu)

        dataset = TensorDataset(input_ids, attention_mask)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        embeddings = []

        with torch.no_grad():
            for batch in dataloader:
                batch_input_ids, batch_attention_mask = [x.to(gpu) for x in batch]

                outputs = self.model(batch_input_ids, batch_attention_mask, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
                batch_emb = last_hidden_state[:, 0, :]
                embeddings.append(batch_emb.cpu())

        return torch.cat(embeddings, dim=0)

    def evaluate(self):
        """
        Perform final evaluation with a linear classifier
        """
        logger.info("Performing final evaluation with linear classifier...")
        self.model.to(gpu)
        # Extract embeddings for all splits
        train_emb = self._encode(train_input_ids.to(gpu), train_attention_mask.to(gpu), batch_size=128).to(gpu)
        val_emb = self._encode(val_input_ids.to(gpu), val_attention_mask.to(gpu), batch_size=128).to(gpu)
        test_emb = self._encode(test_input_ids.to(gpu), test_attention_mask.to(gpu), batch_size=128).to(gpu)

        # Train classifier
        classifier = nn.Linear(self.hidden_size, num_labels).to(gpu)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

        train_labels = labels['train'].to(gpu)
        val_labels = labels['val'].to(gpu)
        test_labels = labels['test'].to(gpu)

        best_val_acc = 0
        best_state = None

        # Early stopping parameters
        patience = 10
        min_delta = 0.0001
        counter = 0
        best_val_loss = float('inf')

        # Train the classifier
        for epoch in range(50):  # More epochs for final classifier training
            # Train on training data
            classifier.train()
            optimizer.zero_grad()
            logits = classifier(train_emb)
            loss = F.cross_entropy(logits, train_labels,
                                  weight=torch.tensor(class_weights, dtype=torch.float32).to(gpu))
            loss.backward()
            optimizer.step()

            # Evaluate on validation data
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(val_emb)
                val_loss = F.cross_entropy(val_logits, val_labels,
                                         weight=torch.tensor(class_weights, dtype=torch.float32).to(gpu))
                val_preds = torch.argmax(val_logits, dim=1).cpu()
                val_acc = accuracy_score(val['label'], val_preds)
                val_f1 = f1_score(val['label'], val_preds, average='macro')
                weighted_f1 = f1_score(val['label'], val_preds, average='weighted')

            logger.info(f"Classifier Training - Epoch {epoch+1}: "
                       f"Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}, "
                       f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f},weightted_f1={weighted_f1:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = classifier.state_dict().copy()
                counter = 0

            # Early stopping check based on validation loss
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
                break

        # Load best classifier
        classifier.load_state_dict(best_state)

        # Final evaluation on test set
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(test_emb)   
                     
            test_preds = torch.argmax(test_logits, dim=1).cpu()
            test_acc = accuracy_score(test['label'], test_preds)
            test_f1 = f1_score(test['label'], test_preds, average='macro')
            if num_labels>5:
                top3_pred_acc = top_k_accuracy_score(test['label'], test_logits.cpu(),k=3)
                top5_pred_acc = top_k_accuracy_score(test['label'], test_logits.cpu(),k=5)
                logger.info(f"Top-3 accuracy score:{top3_pred_acc}")
                logger.info(f"Top-5 accuracy score:{top5_pred_acc}")
            unsupervised_metrics_tracker.create_tsne_plot(0, test_emb.cpu().numpy(), test_preds.cpu().numpy())
        logger.info(f"FINAL EVALUATION RESULTS:")
        logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test F1-Score: {test_f1:.4f}")

        logger.info(f"classification report:{classification_report(test['label'], test_preds)}")

        # Save classifier
        torch.save(classifier.state_dict(), os.path.join(self.ckpt_dir, 'final_classifier.pth'))

        return {
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
        }


### Training ###
optimizer = torch.optim.Adam(model.parameters(), lr=bert_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)

epoch_training_times = []
loss_history = defaultdict(list)
current_epoch = 0

def train_step(engine, batch):
    global model, optimizer, current_epoch
    model = model.to(gpu)
    model.train()
    optimizer.zero_grad()

    (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
    y_true = label.type(torch.long)

    if loss_function == 'cross_entropy':
        loss = F.cross_entropy(model(input_ids, attention_mask), y_true,weight=torch.tensor(class_weights, dtype=torch.float32).to(gpu))
    elif loss_function == 'g-loss':
        loss = losses.predict_lpa(model, input_ids, attention_mask, y_true, iter, sigma, num_labels, gamma, gpu,class_weights)
    elif loss_function == 'scl':
        loss = losses.supervised_contrastive_loss(model, input_ids, attention_mask, y_true, temperature)
    elif loss_function == 'triplet':
        loss_fn = losses.BatchAllTripletLoss(model)
        loss = loss_fn(input_ids, attention_mask, y_true)
    elif loss_function == 'cos-sim':
        loss_fn = losses.BatchCosineSimilarityLoss(model)
        loss = loss_fn(input_ids, attention_mask, y_true)
    else:
        raise ValueError(f"Invalid loss function: {loss_function}")

    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    loss_history[current_epoch].append(train_loss)

    return train_loss

trainer = Engine(train_step)

# Replace the existing evaluators
unsupervised_metrics_tracker = UnsupervisedMetricsTracker(model, ckpt_dir)

@trainer.on(Events.EPOCH_STARTED)
def start_epoch_timer(trainer):
    trainer.state.epoch_start_time = time.time()

@trainer.on(Events.EPOCH_STARTED)
def epoch_started_handler(engine):
    global current_epoch
    current_epoch = engine.state.epoch

@trainer.on(Events.EPOCH_COMPLETED)
def epoch_completed_handler(engine):
    epoch = engine.state.epoch
    # Calculate average loss for this epoch
    avg_loss = np.mean(loss_history[epoch])
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

# Function to plot G-loss trend
def plot_gloss_trend():
    """Plot G-loss values over epochs"""
    if loss_function != 'g-loss':
        print("Warning: Current loss function is not 'g-loss'. Plotting current loss function trend.")
    
    epochs = sorted(loss_history.keys())
    avg_losses = []
    
    for epoch in epochs:
        avg_loss = np.mean(loss_history[epoch])
        avg_losses.append(avg_loss)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_function.upper()} Loss')
    plt.title(f'{loss_function.upper()} Loss vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, f'{loss_function}_loss_trend.png'))
    plt.close()
    
    return epochs, avg_losses

def save_loss_history(filename=os.path.join(ckpt_dir,'loss_history.txt')):
    """Save loss history to a file"""
    with open(filename, 'w') as f:
        for epoch in sorted(loss_history.keys()):
            avg_loss = np.mean(loss_history[epoch])
            f.write(f"Epoch {epoch}: {avg_loss:.6f}\n")
    print(f"Loss history saved to {filename}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    # Calculate and store the training time for this epoch
    epoch_end_time = time.time()
    epoch_training_time = epoch_end_time - trainer.state.epoch_start_time
    epoch_training_times.append(epoch_training_time)
    
    # Log the training time
    logger.info(f"Epoch {trainer.state.epoch} training took {epoch_training_time:.2f} seconds.")

    # Now start the evaluation timer
    eval_start_time = time.time()

    # Evaluate using unsupervised metrics
    result = unsupervised_metrics_tracker.evaluate_epoch(trainer.state.epoch)

    # Calculate evaluation time (optional)
    eval_time = time.time() - eval_start_time
    logger.info(f"Epoch {trainer.state.epoch} evaluation took {eval_time:.2f} seconds.")

    # Save checkpoint if this is the best epoch
    if result['is_best']:
        logger.info("New best model checkpoint saved based on silhouette score.\n")
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': trainer.state.epoch,
                'silhouette_scores': unsupervised_metrics_tracker.silhouette_scores,
                'n_clusters': unsupervised_metrics_tracker.n_clusters_history,
                'epoch_training_times': epoch_training_times  # Save training times
            },
            os.path.join(ckpt_dir, 'best_model.pth')
        )

    # Check for early stopping
    if result['early_stop']:
        logger.info("Early stopping triggered. Terminating training.")
        trainer.terminate()

    # Step the scheduler
    scheduler.step()


# Main training loop
logger.info("Training Start:")
start_time = time.time()
trainer.run(loader['train'], max_epochs=nb_epochs)
end_time = time.time()
logger.info(f"Training took {(end_time - start_time):.2f} seconds.")
plot_gloss_trend()
save_loss_history()
# Plot metrics
unsupervised_metrics_tracker.plot_metrics()

logger.info("Training Complete:")
logger.info("Unsupervised Metrics Summary:")
logger.info(f"Best Silhouette Score: {max(unsupervised_metrics_tracker.silhouette_scores):.4f} (Epoch {unsupervised_metrics_tracker.best_epoch})")
logger.info(f"Final Silhouette Score: {unsupervised_metrics_tracker.silhouette_scores[-1]:.4f}")

# Load the best model for final evaluation
best_model_path = os.path.join(ckpt_dir, 'best_model.pth')
# best_model_path=None
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']} for final evaluation")
else:
    logger.info("No best model checkpoint found. Using the final model for evaluation.")

# Create t-SNE plot from best model embeddings if available
if unsupervised_metrics_tracker.best_embeddings is not None:
    logger.info(f"Creating t-SNE plot for best epoch: {unsupervised_metrics_tracker.best_epoch}")
    unsupervised_metrics_tracker.create_tsne_plot(
        unsupervised_metrics_tracker.best_epoch,
        unsupervised_metrics_tracker.best_embeddings,
        unsupervised_metrics_tracker.best_labels
    )

# Perform final evaluation with classifier
final_evaluator = FinalEvaluator(model, ckpt_dir)
final_results = final_evaluator.evaluate()

logger.info("Final classification metrics after fine-tuning:")
logger.info(f"Validation Accuracy: {final_results['val_acc']:.4f}")
logger.info(f"Test Accuracy: {final_results['test_acc']:.4f}")
logger.info(f"Test F1-Score: {final_results['test_f1']:.4f}")
logger.info(f"Average training time:{np.mean(epoch_training_times)}")
logger.info(f"Total training time:{np.sum(epoch_training_times)}")


