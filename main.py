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

# Import functions from utils.py
import utils

# --- Configuration ---
ENTITY_CONCEPT_FILE = "data/MOOCCube/entities/concept.json"
RELATION_FILE = "data/MOOCCube/relations/prerequisite-dependency.json"
CONCEPT_INFO_FILE = "data/MOOCCube/additional_information/concept_information.json"
EMBEDDING_MODEL_NAME = "LaBSE"
CHECKPOINT_PATH = "checkpoints/best_model.pt"  # Path to save the best model

# Hyperparameters
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
GAT_HEADS = 4
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-5
EPOCHS = 200
SEED = 42
VAL_RATIO = 0.1  # Proportion of edges for validation
TEST_RATIO = 0.1  # Proportion of edges for testing
EVAL_EVERY = 10  # How often to evaluate on validation/test sets


# --- Set Seed ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Model Definitions ---
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(
            hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class LinkPredictor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * in_channels, 128)
        self.lin2 = nn.Linear(128, 1)

    def forward(self, z_src, z_dst):
        x = torch.cat([z_src, z_dst], dim=-1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


# --- Training Epoch Function ---
def train_epoch(model, predictor, data, optimizer, criterion, device):
    model.train()
    predictor.train()
    optimizer.zero_grad()

    # Use training graph edges for message passing
    z = model(data.x.to(device), data.train_pos_edge_index.to(device))

    # Positive edges
    pos_edge_index = data.train_pos_edge_index.to(device)
    pos_pred = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]])

    # Negative edges (Sampled for this epoch)
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,  # Use train edges to avoid sampling known positives
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1),  # Sample same number as positives
        method="sparse",  # Efficient sampling
    ).to(device)
    neg_pred = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])

    # Calculate Loss
    predictions = torch.cat([pos_pred, neg_pred], dim=0)
    labels = (
        torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0)
        .unsqueeze(1)
        .to(device)
    )

    loss = criterion(predictions, labels)

    # Backpropagate
    loss.backward()
    optimizer.step()

    return loss.item()


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    num_nodes, id_to_idx, idx_to_id, concept_details = utils.load_concepts(
        ENTITY_CONCEPT_FILE
    )
    edge_index = utils.load_relations(RELATION_FILE, id_to_idx)

    # 2. Generate Features
    node_features, embedding_dim = utils.generate_node_features(
        CONCEPT_INFO_FILE, concept_details, idx_to_id, num_nodes, EMBEDDING_MODEL_NAME
    )

    # 3. Prepare Data Object
    data = utils.prepare_data_object(node_features, edge_index)

    # 4. Split Data
    data = utils.split_data_for_link_prediction(
        data, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO
    )

    # 5. Setup Device, Models, Optimizer, Criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = GAT(
        in_channels=embedding_dim,  # Use dynamically determined dim
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=OUT_CHANNELS,
        heads=GAT_HEADS,
    ).to(device)

    predictor = LinkPredictor(OUT_CHANNELS).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    # 6. Training Loop
    print("\nStarting training...")
    best_val_auc = 0
    final_test_auc = 0
    best_epoch = 0
    start_total_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        start_epoch_time = time.time()
        loss = train_epoch(model, predictor, data, optimizer, criterion, device)
        epoch_time = time.time() - start_epoch_time

        # Evaluate periodically
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            val_auc = utils.evaluate_model(
                model,
                predictor,
                data,
                data.val_pos_edge_index,
                data.val_neg_edge_index,
                device,
            )
            test_auc = utils.evaluate_model(
                model,
                predictor,
                data,
                data.test_pos_edge_index,
                data.test_neg_edge_index,
                device,
            )

            print(
                f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}, Time: {epoch_time:.2f}s"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc = (
                    test_auc  # Store test AUC corresponding to best val AUC
                )
                best_epoch = epoch
                utils.save_checkpoint(
                    model, predictor, optimizer, epoch, CHECKPOINT_PATH
                )

    total_time = time.time() - start_total_time
    print("\nTraining finished.")
    print(f"Total training time: {total_time:.2f}s")
    print(f"Best Validation AUC: {best_val_auc:.4f} achieved at epoch {best_epoch}")
    print(f"Test AUC corresponding to Best Validation AUC: {final_test_auc:.4f}")

    # 7. Optional: Load and evaluate the best model
    print(f"\nLoading best model from {CHECKPOINT_PATH} for final evaluation...")
    # Re-initialize models before loading state
    model_best = GAT(embedding_dim, HIDDEN_CHANNELS, OUT_CHANNELS, GAT_HEADS).to(device)
    predictor_best = LinkPredictor(OUT_CHANNELS).to(device)
    _ = utils.load_checkpoint(
        CHECKPOINT_PATH, model_best, predictor_best
    )  # Don't need optimizer here

    final_test_auc_loaded = utils.evaluate_model(
        model_best,
        predictor_best,
        data,
        data.test_pos_edge_index,
        data.test_neg_edge_index,
        device,
    )
    print(f"Test AUC with loaded best model: {final_test_auc_loaded:.4f}")
