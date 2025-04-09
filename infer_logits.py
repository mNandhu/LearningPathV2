# infer_logits.py (Modified version of infer.py)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import logging
import os
import gc


ITER_FILE = "logs/margin_iter.txt"
if os.path.exists(ITER_FILE):
    with open(ITER_FILE, "r") as f:
        try:
            iter_num = int(f.read().strip())
        except ValueError:
            iter_num = 1
else:
    iter_num = 6

ITER = str(iter_num - 1)

# --- Configuration (MUST MATCH TRAINING CONFIGURATION) ---
CHECKPOINT_PATH = f"checkpoints/best_model_marginloss{ITER}.pt"
EMBEDDING_MODEL_NAME = "LaBSE"
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 64
GAT_HEADS = 2
GAT_DROPOUT = 0.65
PREDICTOR_DROPOUT = 0.55
SEED = 42
DEVICE_PREFERENCE = "cuda"

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.StreamHandler()],
)

# --- Set Seed ---
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --- Model Definitions (Copied from main.py) ---
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout):
        super().__init__()
        self.dropout_rate = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
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
custom_concepts = [
    # --- Core CS ---
    {
        "id": "intro_python",
        "name": "Introduction to Python",
        "text": "Basic syntax, variables, data types like integers, strings, lists in Python.",
    },
    {
        "id": "python_loops",
        "name": "Python Loops",
        "text": "How to use for and while loops in Python for iteration. Loop control statements like break and continue.",
    },
    {
        "id": "python_functions",
        "name": "Python Functions",
        "text": "Defining and calling functions in Python. Arguments, return values, scope.",
    },
    {
        "id": "data_structures",
        "name": "Data Structures",
        "text": "Overview of common data structures like arrays, linked lists, stacks, queues, trees, and graphs. Focus on abstract concepts.",
    },
    {
        "id": "algorithms",
        "name": "Algorithms Analysis",
        "text": "Analyzing algorithm efficiency using Big O notation. Time and space complexity. Asymptotic analysis.",
    },
    {
        "id": "recursion",
        "name": "Recursion",
        "text": "Understanding recursive functions and how they solve problems by breaking them down. Base cases and recursive steps. Relation to stack.",
    },
    # --- Related CS / Math ---
    {
        "id": "discrete_math",
        "name": "Discrete Mathematics",
        "text": "Mathematical structures that are fundamentally discrete rather than continuous. Includes logic, set theory, graph theory, combinatorics.",
    },
    {
        "id": "linear_algebra",
        "name": "Linear Algebra",
        "text": "Branch of mathematics concerning vector spaces and linear mappings between such spaces. Includes matrices, vectors, determinants.",
    },
    {
        "id": "machine_learning",
        "name": "Machine Learning Basics",
        "text": "Introduction to machine learning concepts. Supervised learning, unsupervised learning, regression, classification.",
    },
    # --- Unrelated ---
    {
        "id": "art",
        "name": "Art History",
        "text": "The academic study of the history and development of painting, sculpture, and the other visual arts.",
    },
    {
        "id": "cooking",
        "name": "Basic Cooking Techniques",
        "text": "Fundamental skills for preparing food, such as chopping, sautéing, boiling, baking. Understanding heat and ingredients.",
    },
    {
        "id": "philosophy",
        "name": "Introduction to Philosophy",
        "text": "Exploring fundamental questions about existence, knowledge, values, reason, mind, and language. Major branches like metaphysics, epistemology, ethics.",
    },
    # --- Semantically Similar / Confusable ---
    {
        "id": "python_syntax",
        "name": "Python Syntax Details",
        "text": "Specific rules governing the structure of well-formed Python programs. Indentation, keywords, operators, comments.",
    },  # Similar to intro_python
    {
        "id": "iteration_concepts",
        "name": "Iteration Concepts",
        "text": "General principles of repeating processes in programming. Loops, iterators, generators as abstract concepts.",
    },  # Similar to python_loops
    {
        "id": "complexity_theory",
        "name": "Computational Complexity Theory",
        "text": "Focuses on classifying computational problems according to their inherent difficulty, and relating those classes to each other. P vs NP problem.",
    },  # More advanced than algorithms analysis
]

# Define KNOWN prerequisite relationships *within this expanded custom set*
# Added a few more plausible links for better context graph
custom_existing_edges = [
    ("intro_python", "python_loops"),
    ("intro_python", "python_functions"),
    ("intro_python", "python_syntax"),  # Syntax is part of intro
    ("python_loops", "python_functions"),
    (
        "python_loops",
        "iteration_concepts",
    ),  # Specific loop knowledge informs general concept
    ("data_structures", "algorithms"),
    ("discrete_math", "data_structures"),  # Often a prereq
    ("discrete_math", "algorithms"),  # Often a prereq
    ("python_functions", "recursion"),
    ("algorithms", "complexity_theory"),  # Analysis often precedes deeper theory
    ("linear_algebra", "machine_learning"),  # Foundational math for ML
    ("intro_python", "machine_learning"),  # Need programming for ML
]

# Define the CANDIDATE edges for prediction (More variety)
custom_candidate_edges = [
    # --- Expected High ---
    ("intro_python", "data_structures"),
    ("discrete_math", "recursion"),  # Graph theory/combinatorics can relate
    ("intro_python", "python_syntax"),  # Should be very high, almost self-evident
    # --- Expected Medium/High ---
    ("data_structures", "machine_learning"),  # DS knowledge helps in ML
    ("python_functions", "machine_learning"),  # Need functions for ML implementation
    (
        "iteration_concepts",
        "algorithms",
    ),  # General iteration understanding helps algo design
    # --- Expected Medium/Low ---
    ("python_syntax", "data_structures"),  # Syntax alone isn't enough for DS concepts
    ("python_loops", "algorithms"),
    ("python_functions", "algorithms"),
    ("data_structures", "recursion"),
    ("algorithms", "recursion"),
    (
        "linear_algebra",
        "algorithms",
    ),  # LA less directly prerequisite than Discrete Math
    # --- Expected Low ---
    ("recursion", "algorithms"),  # Reverse direction
    ("algorithms", "discrete_math"),  # Reverse direction
    ("complexity_theory", "algorithms"),  # Reverse direction
    ("python_loops", "recursion"),
    ("iteration_concepts", "recursion"),  # General iteration vs specific technique
    # --- Expected Very Low (Unrelated) ---
    ("art", "data_structures"),
    ("art", "algorithms"),
    ("art", "machine_learning"),
    ("cooking", "intro_python"),
    ("cooking", "algorithms"),
    ("philosophy", "recursion"),
    ("philosophy", "linear_algebra"),
    # --- Expected Very Low (Related Domain, Wrong Direction/Link) ---
    ("machine_learning", "linear_algebra"),  # Reverse direction
    ("machine_learning", "intro_python"),  # Reverse direction
    # --- Confusable Pairs ---
    ("intro_python", "iteration_concepts"),  # Basic python vs general concept
    ("python_loops", "python_syntax"),  # Specific loops vs general syntax
    (
        "algorithms",
        "complexity_theory",
    ),  # Analysis vs deeper theory (already an edge, check prob)
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

    # 2. Process Custom Data (Same as infer.py)
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
