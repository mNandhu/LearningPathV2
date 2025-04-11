import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
import numpy as np
import random
import time
import logging
import gc
import os
import utils.utils as utils


ITER_FILE = "logs/iter.txt"
if os.path.exists(ITER_FILE):
    with open(ITER_FILE, "r") as f:
        try:
            iter_num = int(f.read().strip())
        except ValueError:
            iter_num = 1
else:
    iter_num = 1

with open(ITER_FILE, "w") as f:
    f.write(str(iter_num + 1))

ITER = str(iter_num)

# --- Configuration ---
# File Paths
ENTITY_CONCEPT_FILE = "data/MOOCCube/entities/concept_formatted.json"
RELATION_FILE = "data/MOOCCube/relations/prerequisite-dependency_formatted.json"
CONCEPT_INFO_FILE = (
    "data/MOOCCube/additional_information/concept_infomation_formatted.json"
)
LOG_FILE = f"logs/training{ITER}.log"
CHECKPOINT_PATH = f"checkpoints/best_model{ITER}.pt"
CACHE_DIR = "cache"  # <-- Directory for cached data
CACHE_FILE_PATH = os.path.join(
    CACHE_DIR, "processed_data_split.pt"
)  # <-- Cache file path

# Model & Embedding
EMBEDDING_MODEL_NAME = "LaBSE"
EMBEDDING_BATCH_SIZE = 1024  # Batch size for sentence transformer encoding
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
GAT_HEADS = 2
GAT_DROPOUT = 0.65  # Dropout rate for GAT layers
PREDICTOR_DROPOUT = 0.55  # Dropout rate for predictor MLP

# Training Process
LEARNING_RATE = 0.001  # <-- Using the adjusted LR from previous recommendation
WEIGHT_DECAY = 0.01  # <-- Using the adjusted WD from previous recommendation
EPOCHS = 300
SEED = 42
VAL_RATIO = 0.1
TEST_RATIO = 0.2
EVAL_EVERY = 2
NEG_SAMPLING_RATIO = 1.0  # Ratio of negative edges to positive edges per epoch

# Optimization & Stopping
LR_SCHEDULER_FACTOR = 0.5  # Factor to reduce LR by
LR_SCHEDULER_PATIENCE = 10  # Epochs to wait before reducing LR if no improvement
EARLY_STOPPING_PATIENCE = 25  # Epochs to wait before stopping if no improvement
EARLY_STOPPING_METRIC = "auc"  # Metric to monitor for scheduler/stopping ('auc' or 'loss') - AUC recommended for link prediction

# --- Setup Logging ---
# (Logging setup remains the same as your last version)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
if not os.path.exists(os.path.dirname(LOG_FILE)):
    try:
        os.makedirs(os.path.dirname(LOG_FILE))
    except OSError as e:
        logging.error(f"Failed to create log directory: {e}", exc_info=True)
        raise
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)

# Log all the configuration parameters
logging.info("Configuration Parameters:")
logging.info(f"  ENTITY_CONCEPT_FILE: {ENTITY_CONCEPT_FILE}")
logging.info(f"  RELATION_FILE: {RELATION_FILE}")
logging.info(f"  CONCEPT_INFO_FILE: {CONCEPT_INFO_FILE}")
logging.info(f"  LOG_FILE: {LOG_FILE}")
logging.info(f"  CHECKPOINT_PATH: {CHECKPOINT_PATH}")
logging.info(f"  CACHE_DIR: {CACHE_DIR}")
logging.info(f"  CACHE_FILE_PATH: {CACHE_FILE_PATH}")
logging.info(f"  EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
logging.info(f"  EMBEDDING_BATCH_SIZE: {EMBEDDING_BATCH_SIZE}")
logging.info(f"  HIDDEN_CHANNELS: {HIDDEN_CHANNELS}")
logging.info(f"  OUT_CHANNELS: {OUT_CHANNELS}")
logging.info(f"  GAT_HEADS: {GAT_HEADS}")
logging.info(f"  GAT_DROPOUT: {GAT_DROPOUT}")
logging.info(f"  PREDICTOR_DROPOUT: {PREDICTOR_DROPOUT}")
logging.info(f"  LEARNING_RATE: {LEARNING_RATE}")
logging.info(f"  WEIGHT_DECAY: {WEIGHT_DECAY}")
logging.info(f"  EPOCHS: {EPOCHS}")
logging.info(f"  SEED: {SEED}")
logging.info(f"  VAL_RATIO: {VAL_RATIO}")
logging.info(f"  TEST_RATIO: {TEST_RATIO}")
logging.info(f"  EVAL_EVERY: {EVAL_EVERY}")
logging.info(f"  NEG_SAMPLING_RATIO: {NEG_SAMPLING_RATIO}")
logging.info(f"  LR_SCHEDULER_FACTOR: {LR_SCHEDULER_FACTOR}")
logging.info(f"  LR_SCHEDULER_PATIENCE: {LR_SCHEDULER_PATIENCE}")
logging.info(f"  EARLY_STOPPING_PATIENCE: {EARLY_STOPPING_PATIENCE}")
logging.info(f"  EARLY_STOPPING_METRIC: {EARLY_STOPPING_METRIC}")
logging.info(f"  ITER: {ITER}")

logging.info("--- Starting New Training Run ---")


# --- Set Seed ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    logging.info(f"PyTorch Seed: {SEED}, CUDA available.")
else:
    logging.info(f"PyTorch Seed: {SEED}, CUDA not available.")


# --- Model Definitions ---
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.dropout_rate = dropout
        self.gat_conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat_conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat_conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.gat_conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
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
        z = model(data.x.to(device), data.train_pos_edge_index.to(device))
        pos_edge_index = data.train_pos_edge_index.to(device)
        pos_pred = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]])
        num_neg_samples = int(pos_edge_index.size(1) * neg_sampling_ratio)
        if num_neg_samples == 0:
            if neg_sampling_ratio > 0:
                num_neg_samples = 1
            else:
                raise ValueError("Negative sampling ratio cannot be zero.")
        if num_neg_samples > 0:
            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=num_neg_samples,
                method="sparse",
            ).to(device)
            neg_pred = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])
        else:
            neg_pred = torch.empty((0, 1), device=device)
        predictions = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat(
            [
                torch.ones(pos_pred.size(0), device=device),
                torch.zeros(neg_pred.size(0), device=device),
            ],
            dim=0,
        ).unsqueeze(1)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        return loss.item()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error("CUDA OOM during training! Skipping epoch.")
            torch.cuda.empty_cache()
            gc.collect()
            return float("inf")
        else:
            logging.error(f"Runtime error during training: {e}", exc_info=True)
            raise
    except Exception as e:
        logging.error(f"Unexpected error during training epoch: {e}", exc_info=True)
        raise


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Initializing ---")

    data = None
    embedding_dim = None
    num_nodes = None
    cache_loaded = False

    # --- Attempt to Load Processed Data from Cache ---
    if os.path.exists(CACHE_FILE_PATH):
        try:
            logging.info(
                f"Attempting to load processed data from cache: {CACHE_FILE_PATH}"
            )
            data = torch.load(CACHE_FILE_PATH, weights_only=False)
            # Verify essential attributes exist
            if (
                hasattr(data, "x")
                and hasattr(data, "train_pos_edge_index")
                and data.num_nodes > 0
            ):
                embedding_dim = data.x.shape[1]
                num_nodes = data.num_nodes
                cache_loaded = True
                logging.info(
                    f"Successfully loaded data from cache. Num nodes: {num_nodes}, Embedding dim: {embedding_dim}"
                )
                logging.info(f"Cached data structure: {data}")
            else:
                logging.warning(
                    "Cached data file is invalid or missing attributes. Re-processing."
                )
                data = None  # Ensure data is None if cache is invalid
        except Exception as e:
            logging.error(
                f"Failed to load data from cache file {CACHE_FILE_PATH}: {e}. Re-processing.",
                exc_info=True,
            )
            data = None  # Ensure data is None if loading fails

    # --- Process Data if Cache Miss ---
    if not cache_loaded:
        logging.info("--- Starting Data Processing (Cache Miss) ---")
        # 1. Load Raw Data
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

        # 3. Prepare Initial Data Object
        initial_data = utils.prepare_data_object(node_features, edge_index)

        # 4. Split Data
        data = utils.split_data_for_link_prediction(
            initial_data, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
        )

        # --- Save Processed Data to Cache ---
        try:
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
                logging.info(f"Created cache directory: {CACHE_DIR}")
            logging.info(f"Saving processed data to cache: {CACHE_FILE_PATH}")
            torch.save(data, CACHE_FILE_PATH)
            logging.info("Successfully saved data to cache.")
        except Exception as e:
            logging.error(f"Failed to save processed data to cache: {e}", exc_info=True)
            # Continue without cache saving, but log the error

        # --- Cleanup ---
        del node_features
        del edge_index
        del concept_details
        del initial_data  # Delete intermediate object
        gc.collect()
        logging.info("Cleaned up intermediate data objects.")

    # --- Ensure data processing/loading was successful ---
    if data is None or embedding_dim is None or num_nodes is None:
        logging.error("Data loading/processing failed. Exiting.")
        exit()

    # 5. Setup Device, Models, Optimizer, Criterion, Scheduler
    device = torch.device(
        "cuda:1" if torch.cuda.is_available() else "cpu"
    )  # Consistent device selection
    logging.info(f"\nUsing device: {device}")

    # Move loaded/processed data object to the target device *once*
    try:
        data = data.to(device)
        logging.info(f"Moved data object to {device}.")
    except Exception as e:
        logging.error(
            f"Failed to move data object to device {device}: {e}", exc_info=True
        )
        exit()

    model = GAT(
        in_channels=embedding_dim,  # Use determined embedding dim
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        heads=GAT_HEADS,
        dropout=GAT_DROPOUT,
    ).to(device)

    predictor = LinkPredictor(
        in_channels=OUT_CHANNELS,
        dropout=PREDICTOR_DROPOUT,
    ).to(device)

    all_params = list(model.parameters()) + list(predictor.parameters())
    optimizer = optim.Adam(all_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if EARLY_STOPPING_METRIC == "auc" else "min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        verbose=True,
    )

    # 6. Training Loop
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
        # Pass data directly, tensors will be moved inside train_epoch if needed (already moved above)
        loss = train_epoch(
            model, predictor, data, optimizer, criterion, device, NEG_SAMPLING_RATIO
        )
        epoch_time = time.time() - start_epoch_time

        if loss == float("inf"):
            logging.warning(f"Epoch {epoch} skipped due to OOM during training.")
            continue

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

            log_msg = (
                f"Epoch: {epoch:03d} | Loss: {loss:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} (P: {val_metrics['precision']:.4f}, R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}) | "
                f"Test AUC: {test_metrics['auc']:.4f} (P: {test_metrics['precision']:.4f}, R: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}) | "
                f"Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            logging.info(log_msg)

            if EARLY_STOPPING_METRIC not in val_metrics:
                raise ValueError(
                    f"Invalid early stopping metric: {EARLY_STOPPING_METRIC}"
                )
            current_val_metric = val_metrics[EARLY_STOPPING_METRIC]
            improved = False
            if EARLY_STOPPING_METRIC == "auc":
                if current_val_metric > best_val_metric:
                    improved = True
            else:
                if current_val_metric < best_val_metric:
                    improved = True

            if improved:
                best_val_metric = current_val_metric
                final_test_metrics = test_metrics
                best_epoch = epoch
                utils.save_checkpoint(
                    model, predictor, optimizer, epoch, best_val_metric, CHECKPOINT_PATH
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler.step(current_val_metric)
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                logging.info(
                    f"Early stopping triggered at epoch {epoch} after {epochs_without_improvement} evaluations without improvement."
                )
                break
        else:
            if epoch % (EVAL_EVERY * 2) == 0:
                logging.info(
                    f"Epoch: {epoch:03d} | Loss: {loss:.4f} | Time: {epoch_time:.2f}s"
                )
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

    logging.info(f"\nLoading best model from {CHECKPOINT_PATH} for final evaluation...")
    try:
        model_best = GAT(
            embedding_dim, HIDDEN_CHANNELS, OUT_CHANNELS, GAT_HEADS, GAT_DROPOUT
        ).to(device)
        predictor_best = LinkPredictor(OUT_CHANNELS, PREDICTOR_DROPOUT).to(device)
        loaded_epoch, _ = utils.load_checkpoint(
            CHECKPOINT_PATH, model_best, predictor_best
        )
        if loaded_epoch > 0:
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
