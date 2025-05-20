import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from docx import Document
import uuid
from datetime import datetime
from redline_parser import process_redline_document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories for storing training data and models
os.makedirs("training_data", exist_ok=True)
os.makedirs("models", exist_ok=True)

class NDADataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    """
    Compute evaluation metrics for the model
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def parse_training_document(file_path):
    """
    Parse a Word document with labeled problematic clauses
    
    Format: Documents should have problematic clauses marked with [PROBLEMATIC] tag
    """
    doc = Document(file_path)
    clauses = []
    labels = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
            
        # Check if the paragraph is marked as problematic
        is_problematic = False
        if "[PROBLEMATIC]" in text:
            is_problematic = True
            text = text.replace("[PROBLEMATIC]", "").strip()
            
        clauses.append(text)
        labels.append(1 if is_problematic else 0)
    
    return clauses, labels

def parse_json_annotations(file_path):
    """
    Parse a JSON file with labeled problematic clauses
    
    Expected format:
    {
        "clauses": [
            {"text": "clause text", "is_problematic": true/false},
            ...
        ]
    }
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    clauses = []
    labels = []
    replacements = []
    
    for clause in data.get("clauses", []):
        clauses.append(clause["text"])
        labels.append(1 if clause.get("is_problematic", False) else 0)
        
        # Check if there's a replacement for this clause
        if clause.get("is_problematic", False) and "replacement" in clause and clause["replacement"]:
            replacements.append({
                "original": clause["text"],
                "replacement": clause["replacement"]
            })
    
    return clauses, labels, replacements

def parse_redline_document(file_path):
    """
    Parse a redline Word document and extract problematic clauses and replacements
    """
    # Process the redline document
    training_data = process_redline_document(file_path)
    
    clauses = []
    labels = []
    replacements = []
    
    for clause in training_data.get("clauses", []):
        clauses.append(clause["text"])
        labels.append(1 if clause.get("is_problematic", False) else 0)
        
        # Check if there's a replacement for this clause
        if clause.get("is_problematic", False) and "replacement" in clause and clause["replacement"]:
            replacements.append({
                "original": clause["text"],
                "replacement": clause["replacement"]
            })
    
    return clauses, labels, replacements

def load_dataset(dataset_id):
    """
    Load all documents in a dataset and prepare for training
    """
    dataset_dir = f"training_data/{dataset_id}"
    
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Dataset {dataset_id} not found")
    
    all_clauses = []
    all_labels = []
    all_replacements = []
    
    # Load dataset metadata
    with open(f"{dataset_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Process each document in the dataset
    for filename in os.listdir(dataset_dir):
        if filename == "metadata.json":
            continue
            
        file_path = os.path.join(dataset_dir, filename)
        
        if filename.endswith('.docx') or filename.endswith('.doc'):
            # Check if it's a redline document
            if "redline" in filename.lower() or metadata.get("is_redline", False):
                clauses, labels, replacements = parse_redline_document(file_path)
                all_replacements.extend(replacements)
            else:
                clauses, labels = parse_training_document(file_path)
                replacements = []
        elif filename.endswith('.json'):
            clauses, labels, replacements = parse_json_annotations(file_path)
            all_replacements.extend(replacements)
        else:
            logger.warning(f"Skipping unsupported file: {filename}")
            continue
            
        all_clauses.extend(clauses)
        all_labels.extend(labels)
    
    logger.info(f"Loaded {len(all_clauses)} clauses from dataset {dataset_id}")
    logger.info(f"Problematic clauses: {sum(all_labels)}, Non-problematic clauses: {len(all_labels) - sum(all_labels)}")
    logger.info(f"Replacements: {len(all_replacements)}")
    
    # Save replacements for future use
    replacements_path = f"{dataset_dir}/replacements.json"
    with open(replacements_path, 'w') as f:
        json.dump(all_replacements, f, indent=2)
    
    return all_clauses, all_labels, metadata

def train_model(dataset_id, epochs=3, batch_size=8, learning_rate=2e-5):
    """
    Train a model on the specified dataset
    """
    # Create a unique job ID
    job_id = str(uuid.uuid4())
    job_dir = f"models/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    
    # Save job metadata
    job_metadata = {
        "id": job_id,
        "dataset_id": dataset_id,
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }
    }
    
    with open(f"{job_dir}/metadata.json", 'w') as f:
        json.dump(job_metadata, f, indent=2)
    
    try:
        # Load the dataset
        clauses, labels, dataset_metadata = load_dataset(dataset_id)
        
        # Split into train and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            clauses, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "nlpaueb/legal-bert-base-uncased", 
            num_labels=2
        )
        
        # Create datasets
        train_dataset = NDADataset(train_texts, train_labels, tokenizer)
        val_dataset = NDADataset(val_texts, val_labels, tokenizer)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=job_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{job_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            learning_rate=learning_rate,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train the model
        logger.info(f"Starting training for job {job_id}")
        trainer.train()
        
        # Evaluate the model
        logger.info(f"Evaluating model for job {job_id}")
        eval_results = trainer.evaluate()
        
        # Save the model
        logger.info(f"Saving model for job {job_id}")
        trainer.save_model(f"{job_dir}/final_model")
        tokenizer.save_pretrained(f"{job_dir}/final_model")
        
        # Update job metadata
        job_metadata["status"] = "completed"
        job_metadata["completed_at"] = datetime.now().isoformat()
        job_metadata["metrics"] = {
            "accuracy": eval_results["eval_accuracy"],
            "precision": eval_results["eval_precision"],
            "recall": eval_results["eval_recall"],
            "f1_score": eval_results["eval_f1"]
        }
        
        with open(f"{job_dir}/metadata.json", 'w') as f:
            json.dump(job_metadata, f, indent=2)
        
        # Create model version
        model_version_id = str(uuid.uuid4())
        model_version = {
            "id": model_version_id,
            "name": f"NDA-Validator-{dataset_metadata['name']}-{datetime.now().strftime('%Y%m%d')}",
            "training_job_id": job_id,
            "dataset_id": dataset_id,
            "created_at": datetime.now().isoformat(),
            "accuracy": eval_results["eval_accuracy"],
            "is_active": False
        }
        
        with open(f"models/versions.json", 'r+') as f:
            try:
                versions = json.load(f)
            except json.JSONDecodeError:
                versions = []
            
            versions.append(model_version)
            
            f.seek(0)
            f.truncate()
            json.dump(versions, f, indent=2)
        
        logger.info(f"Training completed successfully for job {job_id}")
        return job_id, model_version_id
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        
        # Update job metadata to reflect failure
        job_metadata["status"] = "failed"
        job_metadata["error"] = str(e)
        
        with open(f"{job_dir}/metadata.json", 'w') as f:
            json.dump(job_metadata, f, indent=2)
        
        raise

def activate_model(model_version_id):
    """
    Activate a model version for use in the NDA validator
    """
    try:
        with open(f"models/versions.json", 'r+') as f:
            versions = json.load(f)
            
            # Set all models to inactive
            for version in versions:
                version["is_active"] = False
            
            # Set the selected model to active
            for version in versions:
                if version["id"] == model_version_id:
                    version["is_active"] = True
                    active_job_id = version["training_job_id"]
                    break
            else:
                raise ValueError(f"Model version {model_version_id} not found")
            
            # Write back to file
            f.seek(0)
            f.truncate()
            json.dump(versions, f, indent=2)
        
        # Create a symlink to the active model
        active_model_path = f"models/{active_job_id}/final_model"
        if os.path.exists("models/active_model"):
            os.remove("models/active_model")
        os.symlink(active_model_path, "models/active_model", target_is_directory=True)
        
        logger.info(f"Activated model version {model_version_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error activating model: {str(e)}")
        return False

def create_dataset(name, files, is_redline=False):
    """
    Create a new training dataset from uploaded files
    """
    dataset_id = str(uuid.uuid4())
    dataset_dir = f"training_data/{dataset_id}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save dataset metadata
    metadata = {
        "id": dataset_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "document_count": len(files),
        "is_redline": is_redline
    }
    
    with open(f"{dataset_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save the files
    for i, file in enumerate(files):
        file_ext = os.path.splitext(file.filename)[1]
        file_path = os.path.join(dataset_dir, f"document_{i}{file_ext}")
        
        with open(file_path, 'wb') as f:
            f.write(file.file.read())
    
    logger.info(f"Created dataset {dataset_id} with {len(files)} documents")
    return dataset_id, metadata

if __name__ == "__main__":
    # Example usage
    logger.info("This module provides training functionality for the NDA Validator")
    logger.info("It should be imported and used by the main FastAPI application")
