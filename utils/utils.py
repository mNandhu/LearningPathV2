# utils.py
import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import time
import os
import logging  # Use logging module
import gc  # For garbage collection

# Setup basic logging config (can be overridden in main)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_concepts(filepath):
    """Loads concept data, creates mappings."""
    logging.info(f"Loading concepts from: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            concepts_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Concept file not found at {filepath}")
        raise FileNotFoundError(f"Concept file not found at {filepath}")
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {filepath}")
        raise RuntimeError("No valid concepts with IDs found.")

    concept_id_to_index = {}
    index_to_concept_id = {}
    concept_details_by_index = {}
    valid_concepts = 0
    duplicates_found = 0
    for i, concept in enumerate(concepts_data):
        concept_id = concept.get("id")
        if concept_id is None:
            logging.warning(f"Concept at original index {i} is missing 'id'. Skipping.")
            continue
        if concept_id not in concept_id_to_index:
            current_index = len(concept_id_to_index)
            concept_id_to_index[concept_id] = current_index
            index_to_concept_id[current_index] = concept_id
            concept_details_by_index[current_index] = {
                "name": concept.get("name", ""),
                "explanation": concept.get("explanation"),
            }
            valid_concepts += 1
        else:
            # logging.warning(f"Duplicate concept ID '{concept_id}' found. Using first occurrence.") # Can be verbose
            duplicates_found += 1

    if duplicates_found > 0:
        logging.warning(f"Found and ignored {duplicates_found} duplicate concept IDs.")

    num_nodes = len(concept_id_to_index)
    if num_nodes == 0:
        logging.error("No valid concepts with IDs found.")
        raise RuntimeError("No valid concepts with IDs found.")
    logging.info(f"Found {num_nodes} concepts with unique IDs.")
    return num_nodes, concept_id_to_index, index_to_concept_id, concept_details_by_index


def load_relations(filepath, concept_id_to_index):
    """Loads relation data and creates edge_index."""
    logging.info(f"Loading relations from: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            relations_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Relation file not found at {filepath}. Ensure pre-processing.")
        raise RuntimeError(
            "No valid, non-self-loop, unique prerequisite edges found. Check relation file and concept IDs."
        )
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {filepath}. Ensure pre-processing.")
        raise

    edge_list = []
    valid_relations = 0
    invalid_relations = 0
    skipped_self_loops = 0
    duplicates_ignored = 0
    seen_edges = set()

    for rel in relations_data:
        source_id = rel.get("source_concept_id")
        target_id = rel.get("target_concept_id")

        if source_id in concept_id_to_index and target_id in concept_id_to_index:
            source_idx = concept_id_to_index[source_id]
            target_idx = concept_id_to_index[target_id]

            if source_idx == target_idx:
                skipped_self_loops += 1
                continue

            edge_tuple = (source_idx, target_idx)
            if edge_tuple not in seen_edges:
                edge_list.append([source_idx, target_idx])
                seen_edges.add(edge_tuple)
                valid_relations += 1
            else:
                duplicates_ignored += 1
        else:
            invalid_relations += 1

    if not edge_list:
        logging.error(
            "No valid, non-self-loop, unique prerequisite edges found. Check relation file and concept IDs."
        )
        raise RuntimeError(
            "No valid, non-self-loop, unique prerequisite edges found. Check relation file and concept IDs."
        )

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    logging.info(
        f"Found {edge_index.size(1)} valid, unique, non-self-loop prerequisite edges."
    )
    if invalid_relations > 0:
        logging.warning(
            f"Skipped {invalid_relations} relations due to missing concept IDs."
        )
    if skipped_self_loops > 0:
        logging.warning(f"Skipped {skipped_self_loops} self-loop relations.")
    if duplicates_ignored > 0:
        logging.warning(
            f"Ignored {duplicates_ignored} duplicate relations found in file."
        )

    return edge_index


def generate_node_features(
    concept_info_filepath,
    concept_details_by_index,
    index_to_concept_id,
    num_nodes,
    model_name,
    batch_size=128,
):
    """Generates node features using SentenceTransformer and fallback logic."""
    logging.info(f"\nLoading concept information from: {concept_info_filepath}")
    try:
        with open(concept_info_filepath, "r", encoding="utf-8") as f:
            concept_info_data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Concept information file not found at {concept_info_filepath}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {concept_info_filepath}")
        raise

    concept_text_lookup = {}
    for info in concept_info_data:
        name = info.get("name")
        if name:
            text = info.get("wiki_abstract")
            if not text:
                if info.get("baidu_snippet_zh"):
                    text = ""
                    for item in info["baidu_snippet_zh"]:
                        if item.get("snippet"):
                            text += item.get("title") + item["snippet"]
            if text:
                concept_text_lookup[name] = text

    logging.info(
        f"Loaded text information for {len(concept_text_lookup)} concept names."
    )

    texts_to_embed = []
    counts = {"info": 0, "expl": 0, "name": 0, "miss": 0}

    for i in range(num_nodes):
        details = concept_details_by_index.get(i)  # Use .get for safety
        if not details:
            logging.warning(
                f"Missing concept details for index {i}. Using empty string."
            )
            texts_to_embed.append("")
            counts["miss"] += 1
            continue

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
            node_text = ""
            counts["miss"] += 1
            logging.warning(
                f"No text for concept index {i} (ID: {index_to_concept_id.get(i, 'UNKNOWN')}). Using empty string."
            )

        texts_to_embed.append(node_text)

    logging.info(
        f"Text sources: {counts['info']} info, {counts['expl']} explanation, {counts['name']} name, {counts['miss']} missing."
    )

    logging.info(f"\nLoading Sentence Transformer model: {model_name}...")
    start_time = time.time()
    try:
        # Consider specifying cache folder if needed: cache_folder='/path/to/cache'
        embedding_model = SentenceTransformer(model_name, device="cuda:1")
    except Exception as e:
        logging.error(
            f"Error loading Sentence Transformer model '{model_name}': {e}",
            exc_info=True,
        )
        raise

    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    logging.info(
        f"Model loaded. Embedding dimension: {embedding_dim}. Time: {time.time() - start_time:.2f}s"
    )

    logging.info(f"Generating node embeddings (Batch Size: {batch_size})...")
    start_time = time.time()

    # Pre-validate input texts to avoid encoding errors
    logging.info("Pre-validating text inputs...")
    validated_texts = []
    for i, text in enumerate(texts_to_embed):
        # Ensure text is a string
        if not isinstance(text, str):
            logging.warning(
                f"Non-string text at index {i} (type: {type(text).__name__}). Converting to string."
            )
            print(text)
            text = str(text) if text is not None else ""

        validated_texts.append(text)

    try:
        embeddings_np = embedding_model.encode(
            validated_texts, show_progress_bar=True, batch_size=batch_size
        )
    except Exception as e:
        logging.error(f"Error during sentence embedding encoding: {e}", exc_info=True)
        raise RuntimeError(
            "Error during sentence embedding encoding. Check your input data."
        )
    logging.info(
        f"Embedding generation complete. Time: {time.time() - start_time:.2f}s"
    )

    node_features = torch.tensor(embeddings_np, dtype=torch.float)
    logging.info(f"Node feature matrix shape: {node_features.shape}")

    # Explicitly clear large objects from memory
    del embedding_model
    del embeddings_np
    del texts_to_embed
    del concept_text_lookup
    del concept_info_data
    gc.collect()  # Trigger garbage collection

    return node_features, embedding_dim


def prepare_data_object(node_features, edge_index):
    """Creates the PyG Data object."""
    data = Data(x=node_features, edge_index=edge_index)
    logging.info("\nCreated PyG Data object.")
    logging.info(data)
    return data


def split_data_for_link_prediction(data, val_ratio=0.1, test_ratio=0.1):
    """Splits edges for link prediction using PyG utility."""
    logging.info(
        f"Splitting data (Val Ratio: {val_ratio}, Test Ratio: {test_ratio})..."
    )
    data.train_mask = data.val_mask = data.test_mask = None
    if data.num_edges < 20:  # Increased threshold for more reliable splits
        logging.error(
            f"Too few edges ({data.num_edges}) for reliable train/val/test split with ratios {val_ratio}/{test_ratio}."
        )
        raise ValueError(
            f"Too few edges ({data.num_edges}) for reliable train/val/test split with ratios {val_ratio}/{test_ratio}."
        )
    try:
        # train_test_split_edges performs negative sampling for val/test sets by default
        data_split = train_test_split_edges(
            data, val_ratio=val_ratio, test_ratio=test_ratio
        )
        logging.info("Data split complete.")
        logging.info(f"Training edges: {data_split.train_pos_edge_index.size(1)}")
        logging.info(
            f"Validation edges: {data_split.val_pos_edge_index.size(1)} pos / {data_split.val_neg_edge_index.size(1)} neg"
        )
        logging.info(
            f"Test edges: {data_split.test_pos_edge_index.size(1)} pos / {data_split.test_neg_edge_index.size(1)} neg"
        )
        return data_split
    except IndexError as e:
        logging.error(
            f"IndexError during data splitting: {e}. Check edge indices and graph structure.",
            exc_info=True,
        )
        raise
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during data splitting: {e}", exc_info=True
        )
        raise


def save_checkpoint(
    model, predictor, optimizer, epoch, val_auc, filepath="checkpoints/best_model.pt"
):
    """Saves model, predictor, and optimizer state."""
    dir_path = os.path.dirname(filepath)
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logging.info(f"Created checkpoint directory: {dir_path}")
        except OSError as e:
            logging.error(f"Could not create checkpoint directory {dir_path}: {e}")
            return  # Don't try to save if dir creation failed

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_auc": val_auc,  # Store validation AUC for reference
    }
    try:
        torch.save(checkpoint, filepath)
        logging.info(
            f"Checkpoint saved to {filepath} at epoch {epoch} (Val AUC: {val_auc:.4f})"
        )
    except Exception as e:
        logging.error(f"Failed to save checkpoint to {filepath}: {e}")


def load_checkpoint(filepath, model, predictor, optimizer=None):
    """Loads checkpoint into model, predictor, and optionally optimizer."""
    if not os.path.exists(filepath):
        logging.warning(
            f"Checkpoint file not found at {filepath}. Starting from scratch."
        )
        return 0, 0.0  # Return epoch 0, best_val_auc 0.0
    try:
        # Load checkpoint onto the same device model is on
        device = next(model.parameters()).device
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        predictor.load_state_dict(checkpoint["predictor_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logging.info("Loaded optimizer state from checkpoint.")
        else:
            logging.info(
                "Optimizer state not loaded (not provided or not in checkpoint)."
            )
        epoch = checkpoint.get("epoch", 0)
        best_val_auc = checkpoint.get("val_auc", 0.0)
        logging.info(
            f"Checkpoint loaded successfully from {filepath} (Epoch {epoch}, Val AUC: {best_val_auc:.4f})."
        )
        return epoch, best_val_auc
    except Exception as e:
        logging.error(f"Error loading checkpoint from {filepath}: {e}", exc_info=True)
        return 0, 0.0


@torch.no_grad()
def evaluate_model(model, predictor, data, pos_edge_index, neg_edge_index, device):
    """Evaluates the model, returns dict of metrics (AUC, Precision, Recall, F1)."""
    model.eval()
    predictor.eval()
    metrics = {"auc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        # Use training edges for message passing
        z = model(data.x.to(device), data.train_pos_edge_index.to(device))

        # --- Positive Edges ---
        pos_edge_index = pos_edge_index.to(device)
        pos_pred_logits = predictor(z[pos_edge_index[0]], z[pos_edge_index[1]])

        # --- Negative Edges ---
        neg_edge_index = neg_edge_index.to(device)
        neg_pred_logits = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])

        # Combine predictions and labels
        predictions_logits = torch.cat([pos_pred_logits, neg_pred_logits], dim=0)
        labels = torch.cat(
            [torch.ones(pos_pred_logits.size(0)), torch.zeros(neg_pred_logits.size(0))],
            dim=0,
        ).to(device)

        # --- Calculate Metrics ---
        probabilities = torch.sigmoid(predictions_logits).cpu().numpy()
        labels_np = labels.cpu().numpy()
        predicted_labels = (probabilities > 0.5).astype(
            int
        )  # Threshold probabilities for Prec/Rec/F1

        # AUC
        try:
            metrics["auc"] = roc_auc_score(labels_np, probabilities)
        except ValueError:
            logging.warning(
                "AUC calculation failed (likely only one class present in labels). Setting AUC to 0.0."
            )
            metrics["auc"] = 0.0

        # Precision, Recall, F1-score
        # Use average='binary' assuming positive class (link exists) is class '1'
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, predicted_labels, average="binary", zero_division=0
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error("CUDA out of memory during evaluation!")
            # Handle OOM during evaluation (e.g., skip evaluation for this epoch)
            # Returning empty metrics signals an issue
            return metrics  # Return default zero metrics
        else:
            logging.error(f"Runtime error during evaluation: {e}", exc_info=True)
            return metrics  # Return default zero metrics
    except Exception as e:
        logging.error(f"Unexpected error during evaluation: {e}", exc_info=True)
        return metrics  # Return default zero metrics

    return metrics
