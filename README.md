# G-Loss
Graph-based loss function for fine-tuning language models using LPA.

This script implements a full pipeline for fine-tuning transformer models (BERT, RoBERTa, etc.) on text classification tasks using advanced loss functions (G-Loss, Supervised Contrastive Loss, Triplet Loss, Cosine Similarity Loss), with unsupervised evaluation (silhouette score) and final supervised evaluation on test data using a linear classifier.

## Requirements

- Python 
- PyTorch
- transformers
- scikit-learn
- matplotlib
- pandas
- ignite

### To fine-tune a language model (e.g., BERT-base-uncased) using G-Loss on any dataset (e.g., MR), use
```python new-codes/fine_tuning.py \
    --dataset MR \    
    --bert_init bert-base-uncased \
    --loss g-loss \
    --nb_epochs 80 \
    --batch_size 128 \
    --bert_lr 2e-5 \
    --gamma 0.3 \
    --sigma 0.23
```

**Key arguments:**
- `--dataset`: Dataset name (e.g., MR, 20ng, ohsumed, etc.)
- `--bert_init`: Model name (e.g., bert-base-uncased, roberta-base)
- `--loss`: Loss function ( `g-loss`, `scl`, `triplet`, `cos-sim`)
- `--nb_epochs`: Number of epochs
- `--batch_size`: Batch size
- `--bert_lr`: Learning rate for BERT/transformer
- `--gamma`, `--sigma`: Graph loss hyperparameters
- `--temperature`: SCL temperature (if using SCL)
- `--optuna_results`: Path to Optuna best params JSON (optional)
- `--use_latest_checkpoint`: Resume the latest checkpoint if available

---
## Outputs

- **Checkpoints, logs, and plots** are saved in `checkpoint/` directory.
- **Training and evaluation logs:**  
  `checkpoint/<run-folder>/training_finetuning_plotting.log`
- **t-SNE and silhouette plots:**  
  `checkpoint/<run-folder>/`
- **Final classifier weights:**  
  `checkpoint/<run-folder>/final_classifier.pth`
