# utils.py
import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
import time
import os  # For creating directories


def load_concepts(filepath):
    """Loads concept data, creates mappings."""
    print(f"Loading concepts from: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            concepts_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Concept file not found at {filepath}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        exit()

    concept_id_to_index = {}
    index_to_concept_id = {}
    concept_details_by_index = {}
    valid_concepts = 0
    for i, concept in enumerate(concepts_data):
        concept_id = concept.get("id")
        if concept_id is None:
            print(f"Warning: Concept at original index {i} is missing 'id'. Skipping.")
            continue
        # Ensure unique IDs if duplicates exist (take the first encountered)
        if concept_id not in concept_id_to_index:
            current_index = len(concept_id_to_index)  # Use current length as the index
            concept_id_to_index[concept_id] = current_index
            index_to_concept_id[current_index] = concept_id
            concept_details_by_index[current_index] = {
                "name": concept.get("name", ""),
                "explanation": concept.get("explanation"),
            }
            valid_concepts += 1
        else:
            print(
                f"Warning: Duplicate concept ID '{concept_id}' found. Using first occurrence."
            )

    num_nodes = len(concept_id_to_index)
    if num_nodes == 0:
        print("Error: No valid concepts with IDs found.")
        exit()
    print(f"Found {num_nodes} concepts with unique IDs.")
    return num_nodes, concept_id_to_index, index_to_concept_id, concept_details_by_index


def load_relations(filepath, concept_id_to_index):
    """Loads relation data and creates edge_index."""
    print(f"Loading relations from: {filepath}")
    # Assuming pre-processed JSON list of {"source_concept_id": ..., "target_concept_id": ...}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            relations_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Relation file not found at {filepath}. Ensure pre-processing.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. Ensure pre-processing.")
        exit()

    edge_list = []
    valid_relations = 0
    invalid_relations = 0
    skipped_self_loops = 0
    seen_edges = set()  # To handle potential duplicate relations

    for rel in relations_data:
        source_id = rel.get("source_concept_id")
        target_id = rel.get("target_concept_id")

        if source_id in concept_id_to_index and target_id in concept_id_to_index:
            source_idx = concept_id_to_index[source_id]
            target_idx = concept_id_to_index[target_id]

            if source_idx == target_idx:
                skipped_self_loops += 1
                continue  # Skip self-loops for prerequisite graph

            edge_tuple = (source_idx, target_idx)
            if edge_tuple not in seen_edges:
                edge_list.append([source_idx, target_idx])
                seen_edges.add(edge_tuple)
                valid_relations += 1
            else:
                # Silently ignore duplicate relations or add a counter/warning
                pass
        else:
            missing_ids = []
            if source_id not in concept_id_to_index:
                missing_ids.append(f"source '{source_id}'")
            if target_id not in concept_id_to_index:
                missing_ids.append(f"target '{target_id}'")
            # print(f"Warning: Skipped relation due to missing concept ID(s): {', '.join(missing_ids)}") # Can be verbose
            invalid_relations += 1

    if not edge_list:
        print(
            "Error: No valid, non-self-loop, unique prerequisite edges found. Check relation file and concept IDs."
        )
        exit()

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    print(
        f"Found {edge_index.size(1)} valid, unique, non-self-loop prerequisite edges."
    )
    if invalid_relations > 0:
        print(f"Skipped {invalid_relations} relations due to missing concept IDs.")
    if skipped_self_loops > 0:
        print(f"Skipped {skipped_self_loops} self-loop relations.")

    return edge_index


def generate_node_features(
    concept_info_filepath,
    concept_details_by_index,
    index_to_concept_id,
    num_nodes,
    model_name,
):
    """Generates node features using SentenceTransformer and fallback logic."""
    print("\nLoading concept information for embeddings...")
    try:
        with open(concept_info_filepath, "r", encoding="utf-8") as f:
            concept_info_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Concept information file not found at {concept_info_filepath}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {concept_info_filepath}")
        exit()

    concept_text_lookup = {}
    for info in concept_info_data:
        name = info.get("name")
        if name:
            text = info.get("wiki_abstract") or info.get("baidu_snippet_zh")
            if text:
                concept_text_lookup[name] = text

    print(f"Loaded text information for {len(concept_text_lookup)} concept names.")

    texts_to_embed = []
    counts = {"info": 0, "expl": 0, "name": 0, "miss": 0}

    for i in range(num_nodes):
        details = concept_details_by_index[i]
        concept_name = details["name"]
        concept_explanation = details["explanation"]
        node_text = None

        if concept_name in concept_text_lookup:
            node_text = concept_text_lookup[concept_name]
            counts["info"] += 1
        elif concept_explanation:
            node_text = concept_explanation
            counts["expl"] += 1
        elif concept_name:
            node_text = concept_name
            counts["name"] += 1
        else:
            node_text = ""  # Embed an empty string
            counts["miss"] += 1
            print(
                f"Warning: No text for concept index {i} (ID: {index_to_concept_id[i]}). Using empty string."
            )

        texts_to_embed.append(node_text)

    print(
        f"Text sources: {counts['info']} from info file, {counts['expl']} from explanation, {counts['name']} from name, {counts['miss']} missing."
    )

    print(f"\nLoading Sentence Transformer model: {model_name}...")
    start_time = time.time()
    try:
        embedding_model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading Sentence Transformer model '{model_name}': {e}")
        exit()

    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    print(
        f"Model loaded. Embedding dimension: {embedding_dim}. Time: {time.time() - start_time:.2f}s"
    )

    print("Generating node embeddings...")
    start_time = time.time()
    embeddings_np = embedding_model.encode(
        texts_to_embed, show_progress_bar=True, batch_size=128
    )  # Added batch_size
    print(f"Embedding generation complete. Time: {time.time() - start_time:.2f}s")

    node_features = torch.tensor(embeddings_np, dtype=torch.float)
    print(f"Node feature matrix shape: {node_features.shape}")

    del embedding_model  # Free memory
    return node_features, embedding_dim


def prepare_data_object(node_features, edge_index):
    """Creates the PyG Data object."""
    data = Data(x=node_features, edge_index=edge_index)
    print("\nCreated PyG Data object.")
    print(data)
    return data


def split_data_for_link_prediction(data, val_ratio=0.1, test_ratio=0.1):
    """Splits edges for link prediction using PyG utility."""
    print("Splitting data for link prediction...")
    data.train_mask = data.val_mask = data.test_mask = None  # Ensure no node masks
    # Prevent splitting if too few edges exist
    if data.num_edges < 10:  # Arbitrary threshold, adjust if needed
        print(
            f"Error: Too few edges ({data.num_edges}) to perform reliable train/val/test split."
        )
        exit()
    try:
        # Note: This modifies 'data' in place adding split attributes
        data_split = train_test_split_edges(
            data, val_ratio=val_ratio, test_ratio=test_ratio
        )
        print("Data split complete.")
        print(data_split)
        return data_split
    except IndexError as e:
        print(f"\nError during data splitting (IndexError): {e}")
        print("This might happen with very few edges or specific graph structures.")
        print(f"Total edges: {data.num_edges}")
        exit()
    except Exception as e:  # Catch other potential errors
        print(f"\nAn unexpected error occurred during data splitting: {e}")
        exit()


def save_checkpoint(
    model, predictor, optimizer, epoch, filepath="checkpoints/best_model.pt"
):
    """Saves model, predictor, and optimizer state."""
    dir_path = os.path.dirname(filepath)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)  # Create directory if it doesn't exist
        print(f"Created checkpoint directory: {dir_path}")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath} at epoch {epoch}")


def load_checkpoint(filepath, model, predictor, optimizer=None):
    """Loads checkpoint into model, predictor, and optionally optimizer."""
    if not os.path.exists(filepath):
        print(f"Warning: Checkpoint file not found at {filepath}. Cannot load.")
        return 0  # Return epoch 0 if no checkpoint
    try:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictor.load_state_dict(checkpoint["predictor_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state.")
        else:
            print("Optimizer state not loaded (not provided or not in checkpoint).")
        epoch = checkpoint.get("epoch", 0)  # Get epoch if saved, else 0
        print(f"Checkpoint loaded successfully from {filepath} (Epoch {epoch}).")
        return epoch
    except Exception as e:
        print(f"Error loading checkpoint from {filepath}: {e}")
        return 0


@torch.no_grad()
def evaluate_model(model, predictor, data, pos_edge_index, neg_edge_index, device):
    """Evaluates the model on given positive and negative edges, returns AUC."""
    model.eval()
    predictor.eval()

    # Use training edges for message passing
    z = model(data.x.to(device), data.train_pos_edge_index.to(device))

    # Positive edges
    pos_edge_index = pos_edge_index.to(device)
    pos_pred = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]])

    # Negative edges
    neg_edge_index = neg_edge_index.to(device)
    neg_pred = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])

    # Calculate AUC
    predictions = torch.cat([pos_pred, neg_pred], dim=0)
    labels = torch.cat(
        [torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0
    ).to(device)

    probabilities = torch.sigmoid(predictions).cpu().numpy()
    labels_np = labels.cpu().numpy()

    try:
        auc = roc_auc_score(labels_np, probabilities)
    except ValueError as e:
        print(f"Warning during AUC calculation: {e}. Check label distribution.") # Can be verbose
        auc = 0.0  # Return 0 if AUC is undefined (e.g., only one class)

    return auc
