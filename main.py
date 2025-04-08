# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
import numpy as np
import random
import time
import logging  # Use logging module
import gc  # For garbage collection
import os
import utils.utils as utils  # Import functions from utils.py

# --- Configuration ---
# File Paths
ENTITY_CONCEPT_FILE = "data/MOOCCube/entities/concept_formatted.json"
RELATION_FILE = "data/MOOCCube/relations/prerequisite-dependency_formatted.json"
CONCEPT_INFO_FILE = "data/MOOCCube/additional_information/concept_infomation_formatted.json"
LOG_FILE = "logs/training.log"  # File to save logs
CHECKPOINT_PATH = "checkpoints/best_model.pt"

# Model & Embedding
EMBEDDING_MODEL_NAME = "LaBSE"
EMBEDDING_BATCH_SIZE = 128  # Batch size for sentence transformer encoding
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
GAT_HEADS = 4
GAT_DROPOUT = 0.6  # Dropout rate for GAT layers
PREDICTOR_DROPOUT = 0.5  # Dropout rate for predictor MLP

# Training Process
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-5
EPOCHS = 300  # Increased epochs, rely on early stopping
SEED = 42
VAL_RATIO = 0.1
TEST_RATIO = 0.1
EVAL_EVERY = 5  # Evaluate more frequently
NEG_SAMPLING_RATIO = 1.0  # Ratio of negative edges to positive edges per epoch

# Optimization & Stopping
LR_SCHEDULER_FACTOR = 0.5  # Factor to reduce LR by
LR_SCHEDULER_PATIENCE = 10  # Epochs to wait before reducing LR if no improvement
EARLY_STOPPING_PATIENCE = 30  # Epochs to wait before stopping if no improvement
EARLY_STOPPING_METRIC = "auc"  # Metric to monitor for scheduler/stopping ('auc' or 'loss') - AUC recommended for link prediction

# --- Setup Logging ---
# Remove previous handlers if any
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Ensure log path exists
if not os.path.exists(os.path.dirname(LOG_FILE)):
    try:
        os.makedirs(os.path.dirname(LOG_FILE))
    except OSError as e:
        logging.error(f"Failed to create log directory: {e}", exc_info=True)
        raise
# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),  # Overwrite log file each run
        logging.StreamHandler(),  # Log to console
    ],
)
logging.info("--- Starting New Training Run ---")

# --- Set Seed ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Comment out deterministic for potential speedup if exact reproducibility isn't paramount
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # False for reproducibility focus
    logging.info(f"PyTorch Seed: {SEED}, CUDA available.")
else:
    logging.info(f"PyTorch Seed: {SEED}, CUDA not available.")


# --- Model Definitions ---
class GAT(nn.Module):
    # Added dropout parameter
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.dropout_rate = dropout
        # dropout in GATConv applies to attention weights, additional dropout needed on features
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x, edge_index):
        # Apply feature dropout before first layer
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # Apply feature dropout before second layer
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
    # Added dropout parameter
    def __init__(self, in_channels, dropout):
        super().__init__()
        self.dropout_rate = dropout
        self.lin1 = nn.Linear(2 * in_channels, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, z_src, z_dst):
        x = torch.cat([z_src, z_dst], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x


# --- Training Epoch Function ---
def train_epoch(
    model, predictor, data, optimizer, criterion, device, neg_sampling_ratio
):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    try:
        # Use training graph edges for message passing
        z = model(data.x.to(device), data.train_pos_edge_index.to(device))

        # Positive edges
        pos_edge_index = data.train_pos_edge_index.to(device)
        pos_pred = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]])

        # Negative edges (Sampled for this epoch)
        num_neg_samples = int(pos_edge_index.size(1) * neg_sampling_ratio)
        if (
            num_neg_samples == 0
        ):  # Ensure at least one negative sample if ratio is very low but > 0
            if neg_sampling_ratio > 0:
                num_neg_samples = 1
            else:
                logging.error(
                    "Negative sampling ratio is 0. This configuration is not supported."
                )
                raise ValueError(
                    "Negative sampling ratio cannot be zero. Please set a positive value."
                )

        if num_neg_samples > 0:
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,  # Use train edges to avoid sampling known positives
                num_nodes=data.num_nodes,
                num_neg_samples=num_neg_samples,
                method="sparse",
            ).to(device)
            neg_pred = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])
        else:
            # Handle the case where no negative samples are needed/generated
            neg_pred = torch.empty((0, 1), device=device)  # Empty tensor

        # Calculate Loss
        predictions = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat(
            [
                torch.ones(pos_pred.size(0), device=device),
                torch.zeros(neg_pred.size(0), device=device),
            ],
            dim=0,
        ).unsqueeze(1)  # Ensure labels have shape [N, 1] for BCEWithLogitsLoss

        loss = criterion(predictions, labels)

        # Backpropagate
        loss.backward()
        # Optional: Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error("CUDA out of memory during training! Skipping epoch.")
            torch.cuda.empty_cache()  # Attempt to free memory
            gc.collect()
            return float("inf")  # Return infinity loss to indicate failure
        else:
            logging.error(f"Runtime error during training: {e}", exc_info=True)
            raise  # Re-raise other runtime errors
    except Exception as e:
        logging.error(f"Unexpected error during training epoch: {e}", exc_info=True)
        raise  # Re-raise other exceptions


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Initializing ---")
    # 1. Load Data
    num_nodes, id_to_idx, idx_to_id, concept_details = utils.load_concepts(
        ENTITY_CONCEPT_FILE
    )
    edge_index = utils.load_relations(RELATION_FILE, id_to_idx)

    # 2. Generate Features
    node_features, embedding_dim = utils.generate_node_features(
        CONCEPT_INFO_FILE,
        concept_details,
        idx_to_id,
        num_nodes,
        EMBEDDING_MODEL_NAME,
        EMBEDDING_BATCH_SIZE,
    )

    # 3. Prepare Data Object
    data = utils.prepare_data_object(node_features, edge_index)

    # 4. Split Data
    data = utils.split_data_for_link_prediction(
        data, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
    )

    # --- Cleanup large intermediate objects ---
    del node_features
    del edge_index
    del concept_details
    gc.collect()
    logging.info("Cleaned up intermediate data objects.")
    # ---

    # 5. Setup Device, Models, Optimizer, Criterion, Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"\nUsing device: {device}")

    model = GAT(
        in_channels=embedding_dim,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        heads=GAT_HEADS,
        dropout=GAT_DROPOUT,  # Pass configured dropout
    ).to(device)

    predictor = LinkPredictor(
        in_channels=OUT_CHANNELS,
        dropout=PREDICTOR_DROPOUT,  # Pass configured dropout
    ).to(device)

    # Combine parameters for optimizer
    all_params = list(model.parameters()) + list(predictor.parameters())
    optimizer = optim.Adam(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max"
        if EARLY_STOPPING_METRIC == "auc"
        else "min",  # Monitor max AUC or min loss
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        verbose=True,
    )

    # 6. Training Loop with Enhancements
    logging.info(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    logging.info(f"  Device: {device}")
    logging.info(f"  Optimizer: Adam (LR={LEARNING_RATE}, WD={WEIGHT_DECAY})")
    logging.info(
        f"  Scheduler: ReduceLROnPlateau (Factor={LR_SCHEDULER_FACTOR}, Patience={LR_SCHEDULER_PATIENCE})"
    )
    logging.info(
        f"  Early Stopping: Patience={EARLY_STOPPING_PATIENCE} (Metric: Val {EARLY_STOPPING_METRIC.upper()})"
    )
    logging.info(f"  Negative Sampling Ratio: {NEG_SAMPLING_RATIO}")

    best_val_metric = 0.0 if EARLY_STOPPING_METRIC == "auc" else float("inf")
    final_test_metrics = {}
    best_epoch = 0
    epochs_without_improvement = 0
    start_total_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        start_epoch_time = time.time()
        loss = train_epoch(
            model, predictor, data, optimizer, criterion, device, NEG_SAMPLING_RATIO
        )
        epoch_time = time.time() - start_epoch_time

        if loss == float("inf"):  # Check if OOM occurred in training
            logging.warning(f"Epoch {epoch} skipped due to OOM during training.")
            # Optionally implement logic like reducing batch size (not applicable here) or stopping
            continue  # Skip evaluation if training failed

        # Evaluate periodically
        if epoch % EVAL_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            val_metrics = utils.evaluate_model(
                model,
                predictor,
                data,
                data.val_pos_edge_index,
                data.val_neg_edge_index,
                device,
            )
            test_metrics = utils.evaluate_model(
                model,
                predictor,
                data,
                data.test_pos_edge_index,
                data.test_neg_edge_index,
                device,
            )

            # Log detailed metrics
            log_msg = (
                f"Epoch: {epoch:03d} | Loss: {loss:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} (P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}) | "
                f"Test AUC: {test_metrics['auc']:.4f} (P: {test_metrics['precision']:.4f}, R: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}) | "
                f"Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            logging.info(log_msg)

            # --- Check for Improvement & Early Stopping ---
            # Explicitly check if the specified metric exists in val_metrics
            if EARLY_STOPPING_METRIC not in val_metrics:
                logging.error(
                    f"Specified metric '{EARLY_STOPPING_METRIC}' not found in validation metrics. Available metrics: {list(val_metrics.keys())}"
                )
                raise ValueError(
                    f"Invalid early stopping metric: {EARLY_STOPPING_METRIC}"
                )

            current_val_metric = val_metrics[EARLY_STOPPING_METRIC]
            improved = False
            if EARLY_STOPPING_METRIC == "auc":
                if current_val_metric > best_val_metric:
                    improved = True
            else:  # Assuming 'loss' - though loss isn't calculated in eval currently
                # Placeholder if loss was added to eval_metrics
                if current_val_metric < best_val_metric:
                    improved = True

            if improved:
                best_val_metric = current_val_metric
                final_test_metrics = (
                    test_metrics  # Store test metrics corresponding to best val epoch
                )
                best_epoch = epoch
                utils.save_checkpoint(
                    model, predictor, optimizer, epoch, best_val_metric, CHECKPOINT_PATH
                )
                epochs_without_improvement = 0  # Reset counter
            else:
                epochs_without_improvement += (
                    1  # Increment by 1 per evaluation, not by EVAL_EVERY
                )

            # --- Learning Rate Scheduling ---
            scheduler.step(
                current_val_metric
            )  # Step scheduler based on validation metric

            # --- Early Stopping ---
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                logging.info(
                    f"Early stopping triggered at epoch {epoch} after {EARLY_STOPPING_PATIENCE} epochs without improvement."
                )
                break
        else:
            # Minimal logging for non-evaluation epochs
            if (
                epoch % (EVAL_EVERY * 2) == 0
            ):  # Log loss less frequently between evaluations
                logging.info(
                    f"Epoch: {epoch:03d} | Loss: {loss:.4f} | Time: {epoch_time:.2f}s"
                )

        # Trigger garbage collection periodically (might help on some systems)
        if epoch % 20 == 0:
            gc.collect()

    total_time = time.time() - start_total_time
    logging.info("\n--- Training Finished ---")
    logging.info(f"Total training time: {total_time:.2f}s")
    logging.info(
        f"Best Validation {EARLY_STOPPING_METRIC.upper()}: {best_val_metric:.4f} achieved at epoch {best_epoch}"
    )
    logging.info("Test Metrics @ Best Validation Epoch:")
    for key, value in final_test_metrics.items():
        logging.info(f"  - Test {key.capitalize()}: {value:.4f}")

    # 7. Optional: Load and evaluate the best model explicitly
    logging.info(f"\nLoading best model from {CHECKPOINT_PATH} for final evaluation...")
    try:
        # Re-initialize models before loading state
        model_best = GAT(
            embedding_dim, HIDDEN_CHANNELS, OUT_CHANNELS, GAT_HEADS, GAT_DROPOUT
        ).to(device)
        predictor_best = LinkPredictor(OUT_CHANNELS, PREDICTOR_DROPOUT).to(device)
        loaded_epoch, _ = utils.load_checkpoint(
            CHECKPOINT_PATH, model_best, predictor_best
        )

        if loaded_epoch > 0:  # Check if loading was successful
            final_test_metrics_loaded = utils.evaluate_model(
                model_best,
                predictor_best,
                data,
                data.test_pos_edge_index,
                data.test_neg_edge_index,
                device,
            )
            logging.info(
                f"Final Test Metrics with loaded best model (from epoch {loaded_epoch}):"
            )
            for key, value in final_test_metrics_loaded.items():
                logging.info(f"  - Test {key.capitalize()}: {value:.4f}")
        else:
            logging.warning("Could not load best model for final evaluation.")

    except Exception as e:
        logging.error(
            f"Error during final evaluation of loaded model: {e}", exc_info=True
        )

    logging.info("--- Run Completed ---")
