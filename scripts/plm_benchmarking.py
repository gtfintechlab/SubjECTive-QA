import os
import sys
from time import time, sleep
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, RobertaTokenizerFast, FlaubertTokenizerFast, RobertaForSequenceClassification, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, XLNetForSequenceClassification, XLNetTokenizerFast
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from datasets import load_dataset

def fine_tune_plm(gpu_numbers: str, seed: int, language_model_to_use: str, batch_size: int, learning_rate: float, feature: str, save_model_path: str):
    """
    Fine-tunes a pre-trained language model (PLM) for sequence classification.
    Args:
        gpu_numbers (str): Comma-separated string of GPU numbers to use.
        seed (int): Random seed for reproducibility.
        language_model_to_use (str): The pre-trained language model to use. Options include 'bert-base-uncased', 'yiyanghkust/finbert-pretrain', 'roberta', and 'roberta-large'.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the optimizer.
        feature (str): The target feature/label column in the dataset.
        save_model_path (str): Path to save the fine-tuned model and tokenizer.
    Returns:
        list: A list containing experiment results including seed, learning rate, batch size, best cross-entropy loss, best accuracy, best F1 score, test cross-entropy loss, test accuracy, test F1 score, training time taken, and testing time taken.
    Raises:
        ValueError: If an unsupported language model is specified.
    """
    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_numbers
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load from Hugging Face
    train_dataset = pd.read_csv("train.csv")
    val_dataset = pd.read_csv("val.csv")
    test_dataset = pd.read_csv("test.csv")

    # Tokenizer
    
    #tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base' if language_model_to_use == 'roberta' else 'roberta-large', do_lower_case=True)
    try:
        if language_model_to_use == 'bert-base-uncased':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        elif language_model_to_use == 'yiyanghkust/finbert-pretrain':
            tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain', do_lower_case=True)
        elif language_model_to_use == 'roberta':
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True)
        elif language_model_to_use == 'roberta-large':
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large', do_lower_case=True)
        else:
            raise ValueError("Unsupported language model.")
    except Exception as e:
        print(e)
    
    # Preprocess HF data
    def preprocess_data(dataset):
        concatenated_texts = [q + " " + a for q, a in zip(dataset['QUESTION'], dataset['ANSWER'])]
        labels = dataset[feature]
        tokens = tokenizer(concatenated_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        return TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.LongTensor(labels))

    train_data = preprocess_data(train_dataset)
    val_data = preprocess_data(val_dataset)
    test_data = preprocess_data(test_dataset)

    # Dataloaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Seed setup
    torch.manual_seed(seed)

    # Model setup

    #model = RobertaForSequenceClassification.from_pretrained('roberta-base' if language_model_to_use == 'roberta' else 'roberta-large', num_labels=3)
    #model.to(device)
    try:
        if language_model_to_use == 'bert-base-uncased':
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)
        elif language_model_to_use == 'yiyanghkust/finbert-pretrain':
            model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-pretrain', num_labels=3).to(device)
        elif language_model_to_use == 'roberta':
            model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3).to(device)
        elif language_model_to_use == 'roberta-large':
            model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3).to(device)
        else:
            raise ValueError("Unsupported language model.")
    except Exception as e:
        print(e)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    max_num_epochs = 20
    max_early_stopping = 7
    early_stopping_count = 0
    best_ce = float('inf')
    best_accuracy = float('-inf')
    best_f1 = float('-inf')

    # Training and validation loop
    start_fine_tuning = time()
    for epoch in range(max_num_epochs):
        if early_stopping_count >= max_early_stopping:
            break

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            total_loss, total_accuracy, total_f1 = 0, 0, 0
            for batch in (train_dataloader if phase == 'train' else val_dataloader):
                batch = [b.to(device) for b in batch]
                inputs, masks, labels = batch
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, attention_mask=masks, labels=labels)
                    loss = outputs.loss
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).flatten()
                total_loss += loss.item()
                batch_accuracy = accuracy_score(labels.cpu(), preds.cpu())
                total_accuracy += batch_accuracy
                total_f1 += f1_score(labels.cpu(), preds.cpu(), average='weighted')

            # Calculate average metrics
            avg_loss = total_loss / len(train_dataloader if phase == 'train' else val_dataloader)
            avg_accuracy = total_accuracy / len(train_dataloader if phase == 'train' else val_dataloader)
            avg_f1 = total_f1 / len(train_dataloader if phase == 'train' else val_dataloader)

            # Early stopping and best model logic
            if phase == 'val':
                if avg_loss < best_ce:
                    best_ce = avg_loss
                    best_accuracy = avg_accuracy
                    best_f1 = avg_f1
                    early_stopping_count = 0
                    torch.save({'model_state_dict': model.state_dict()}, 'best_model.pt')
                else:
                    early_stopping_count += 1

            # Print metrics
            print(f"Epoch {epoch + 1} / {max_num_epochs} - Phase: {phase}")
            print(f"Loss: {avg_loss}, Accuracy: {avg_accuracy}, F1 Score: {avg_f1}")
            print(f"Best CE: {best_ce}, Best Accuracy: {best_accuracy}, Best F1 Score: {best_f1}")
            print(f"Early Stopping Counter: {early_stopping_count}")

    training_time_taken = (time() - start_fine_tuning) / 60.0
    print(f"Training time taken: {training_time_taken} minutes")

    # Load best model
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test phase
    start_test_labeling = time()
    model.eval()
    test_ce, test_accuracy, test_f1 = 0, 0, 0
    with torch.no_grad():
        for batch in test_dataloader:
            batch = [b.to(device) for b in batch]
            inputs, masks, labels = batch
            outputs = model(inputs, attention_mask=masks, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).flatten()
            test_ce += outputs.loss.item()
            test_accuracy += accuracy_score(labels.cpu(), preds.cpu())
            test_f1 += f1_score(labels.cpu(), preds.cpu(), average='weighted')

    # Print test metrics
    test_ce /= len(test_dataloader)
    test_accuracy /= len(test_dataloader)
    test_f1 /= len(test_dataloader)
    test_time_taken = (time() - start_test_labeling) / 60.0
    print("Test Metrics:")
    print(f"Test CE: {test_ce}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1}")
    print(f"Testing time taken: {test_time_taken} minutes")

    experiment_results = [seed, learning_rate, batch_size, best_ce, best_accuracy, best_f1, test_ce, test_accuracy, test_f1, training_time_taken, test_time_taken]

    if save_model_path:
        model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)

    return experiment_results


def train_lm_experiments(gpu_numbers: str, train_data_path_prefix: str, test_data_path_prefix: str, language_model_to_use: str, feature: str):
    """
    Conducts a series of experiments to fine-tune a pre-trained language model (PLM) using different seeds, batch sizes, and learning rates.
    Args:
        gpu_numbers (str): Comma-separated string of GPU numbers to use for training.
        train_data_path_prefix (str): Prefix for the training data file paths. The seed will be appended to this prefix to form the complete path.
        test_data_path_prefix (str): Prefix for the test data file paths. The seed will be appended to this prefix to form the complete path.
        language_model_to_use (str): The name of the pre-trained language model to fine-tune.
        feature (str): The specific feature or task for which the model is being fine-tuned.
    Returns:
        None: The function saves the results of the experiments to an Excel file.
    """
    results = []
    seeds = [5768, 78516, 944601]
    batch_sizes = [32, 16, 8, 4]
    learning_rates = [1e-4, 1e-5, 1e-6, 1e-7]
    count = 0
    
    # TODO: ADD THESE AS PARAMETERS FOR FUNCTION CALL IN LINE 170
    train_data_path = train_data_path_prefix + "-" + str(seed) + ".xlsx"
    test_data_path = test_data_path_prefix + "-" + str(seed) + ".xlsx"

    for seed in seeds:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                count += 1
                print(f'Experiment {count} of {len(seeds) * len(batch_sizes) * len(learning_rates)}:')
                results.append(fine_tune_plm(gpu_numbers, str(seed), language_model_to_use, batch_size, learning_rate, feature, None))
                df = pd.DataFrame(results, columns=["Seed", "Learning Rate", "Batch Size", "Val Cross Entropy", "Val Accuracy", "Val F1 Score", "Test Cross Entropy", "Test Accuracy", "Test F1 Score", "Fine Tuning Time(m)", "Test Labeling Time(m)"])
                df.to_excel(f'../../data/results/final_{language_model_to_use}.xlsx', index=False)


if __name__ == '__main__':
    
    features = ["CLEAR", "ASSERTIVE", "CAUTIOUS", "OPTIMISTIC", "SPECIFIC", "RELEVANT"]
    start_t = time()
    
    
    
    # run experiments
    for feature in features:
        for language_model_to_use in ["roberta", "bert-base-uncased", "yiyanghkust/finbert-pretrain"]: # TODO: add the other models here, idk what they're specifically called
            train_data_path_prefix = "../../data/train/" + language_model_to_use +"-" + feature + "-train"
            test_data_path_prefix = "../../data/test/" + language_model_to_use + "-" + feature + "-test"
            train_lm_experiments(gpu_numbers="0",train_data_path_prefix = train_data_path_prefix, test_data_path_prefix=test_data_path_prefix, language_model_to_use=language_model_to_use, feature=feature)
            
        print((time() - start_t) / 60.0)