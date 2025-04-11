import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
# from utils.custom import GATConv
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import logging
import os
import gc
import json


ITER_FILE = "logs/iter.txt"
if os.path.exists(ITER_FILE):
    with open(ITER_FILE, "r") as f:
        try:
            iter_num = int(f.read().strip())
        except ValueError:
            iter_num = 2
else:
    iter_num = 2

ITER = str(iter_num - 1)
# ITER = 8
LOG_FILE = f"logs/training{ITER}.log"
# --- Configuration (MUST MATCH TRAINING CONFIGURATION) ---
CHECKPOINT_PATH = f"checkpoints/best_model{ITER}.pt"
EMBEDDING_MODEL_NAME = "LaBSE"

HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
GAT_HEADS = 2
GAT_DROPOUT = 0.65  # Dropout rate for GAT layers
PREDICTOR_DROPOUT = 0.55  # Dropout rate for predictor MLP

SEED = 42
DEVICE_PREFERENCE = "cuda:1"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="a"),logging.StreamHandler()],
)

# --- Set Seed ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
        return x  # Return raw logits


# --- Custom Data Definition ---
concepts_path = "data/output_testing_zh.json"
logging.info(f"Loading custom concepts from {concepts_path}...")
with open(concepts_path, "r") as f:
    custom_concepts = json.load(f)


# Define KNOWN prerequisite relationships *within this expanded custom set*
# --- Custom Data Definition (Revised for Presentation) ---


# Define KNOWN prerequisite relationships *within this custom set*
# Refined set based on the new concepts
custom_existing_edges = [
    ("intro_python", "data_structures"),
    ("data_structures", "algorithms"),          # Added standard CS link
    ("calculus", "classical_mechanics"),
    ("discrete_math", "algorithms"),           # Added standard CS link
    ("probability", "statistics"),             # Added standard Math link
    ("probability", "machine_learning"),       # Added standard ML link
    ("statistics", "machine_learning"),        # Added standard ML link
    ("statistics", "macroeconomics"),        # Added standard ML link
    ("molecular_biology", "genetics"),         # Added standard Bio link
    ("linear_algebra", "machine_learning"),
    ("intro_python", "machine_learning"),
    ("genetics", "molecular_biology"),         # Plausible Bio link
    # Keep LA -> DS from before
    ("linear_algebra", "data_structures"),
]

# Define the CANDIDATE edges for presentation (Handpicked for clarity)
custom_candidate_edges = [
    # --- Expected High (Clear Prereqs) ---
    ("intro_python", "algorithms"),              # Need programming basics for algos
    ("discrete_math", "data_structures"),      # Very common prereq
    ("calculus", "linear_algebra"),            # Common math sequence

    # --- Expected Medium/High ---
    ("data_structures", "machine_learning"),     # Should be reasonably high

    # --- Expected Medium (Plausible but less direct) ---
    ("classical_mechanics", "linear_algebra"),

    # --- Expected Low (Reverse Directions) ---
    ("algorithms", "data_structures"),
    ("machine_learning", "statistics"),
    ("evolution", "genetics"),             
    ("classical_mechanics", "calculus"),

    # --- Expected Very Low (Clearly Unrelated Domains) ---
    ("music_theory", "algorithms"),
    ("world_war_1", "genetics"),
    ("macroeconomics", "molecular_biology"),

    ("intro_python", "statistics"),
]

# --- Main Inference Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Inference (Logits) ---")

    # 1. Setup Device
    if torch.cuda.is_available() and DEVICE_PREFERENCE.startswith("cuda"):
        device = torch.device(DEVICE_PREFERENCE)
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # 2. Process Custom Data
    logging.info("Processing custom concept data...")
    custom_id_to_index = {concept["id"]: i for i, concept in enumerate(custom_concepts)}
    custom_index_to_id = {i: concept["id"] for i, concept in enumerate(custom_concepts)}
    num_custom_nodes = len(custom_concepts)
    custom_texts = [concept["text"] for concept in custom_concepts]

    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        embedding_dim = embedding_model.get_sentence_embedding_dimension()
        logging.info(f"Embedding model loaded. Dimension: {embedding_dim}")
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}", exc_info=True)
        exit()

    logging.info("Generating embeddings for custom concepts...")
    try:
        embeddings_np = embedding_model.encode(custom_texts, show_progress_bar=False)
        custom_node_features = torch.tensor(embeddings_np, dtype=torch.float).to(device)
        logging.info(f"Generated custom node features: {custom_node_features.shape}")
    except Exception as e:
        logging.error(f"Failed to generate embeddings: {e}", exc_info=True)
        exit()

    custom_edge_list = []
    for src_id, dst_id in custom_existing_edges:
        if src_id in custom_id_to_index and dst_id in custom_id_to_index:
            src_idx, dst_idx = custom_id_to_index[src_id], custom_id_to_index[dst_id]
            if src_idx != dst_idx:
                custom_edge_list.append([src_idx, dst_idx])
    if not custom_edge_list:
        custom_edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
    else:
        custom_edge_index = (
            torch.tensor(custom_edge_list, dtype=torch.long).t().contiguous().to(device)
        )
    logging.info(f"Created custom edge index with {custom_edge_index.size(1)} edges.")

    # 3. Initialize Models
    logging.info("Initializing GAT and LinkPredictor models...")
    model = GAT(
        embedding_dim, HIDDEN_CHANNELS, OUT_CHANNELS, GAT_HEADS, GAT_DROPOUT
    ).to(device)
    predictor = LinkPredictor(OUT_CHANNELS, PREDICTOR_DROPOUT).to(device)

    # 4. Load Checkpoint
    logging.info(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        logging.error("Checkpoint file not found.")
        exit()
    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        predictor.load_state_dict(checkpoint["predictor_state_dict"])
        loaded_epoch = checkpoint.get("epoch", "N/A")
        loaded_val_auc = checkpoint.get("val_auc", "N/A")
        logging.info(
            f"Checkpoint loaded successfully (epoch {loaded_epoch}, Val AUC {loaded_val_auc:.4f})."
        )
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}", exc_info=True)
        exit()

    # 5. Perform Inference - GET LOGITS
    logging.info("\n--- Predicting LOGITS for Candidate Edges ---")
    model.eval()
    predictor.eval()
    results = []
    with torch.no_grad():
        try:
            z = model(custom_node_features, custom_edge_index)
            for src_id, dst_id in custom_candidate_edges:
                if src_id in custom_id_to_index and dst_id in custom_id_to_index:
                    src_idx, dst_idx = (
                        custom_id_to_index[src_id],
                        custom_id_to_index[dst_id],
                    )
                    z_src, z_dst = z[src_idx], z[dst_idx]
                    logit = (
                        predictor(z_src.unsqueeze(0), z_dst.unsqueeze(0))
                        .squeeze()
                        .item()
                    )  # Get logit value
                    probability = torch.sigmoid(
                        torch.tensor(logit)
                    ).item()  # Calculate probability for reference
                    results.append(
                        {
                            "source": src_id,
                            "target": dst_id,
                            "logit": logit,
                            "probability": probability,
                        }
                    )
                else:
                    logging.warning(
                        f"Skipping candidate edge with unknown ID: {src_id} -> {dst_id}"
                    )
        except Exception as e:
            logging.error(f"Error during inference: {e}", exc_info=True)

    # 6. Display Results
    print("\n--- Inference Results (Logits & Probabilities) ---")
    if results:
        results.sort(key=lambda x: x["probability"], reverse=True)
        print(
            f"{'Source Concept':<20} -> {'Target Concept':<20} | {'LOGIT':<15} | {'Probability':<25}"
        )
        print("-" * 80)
        for res in results:
            print(
                f"{res['source']:<20} -> {res['target']:<20} | {res['logit']:<15.4f} | {res['probability']:.4f}"
            )
    else:
        print("No predictions could be made.")

    logging.info("--- Inference Completed ---")

    # Optional: Clean up
    del model, predictor, embedding_model, custom_node_features, z
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
