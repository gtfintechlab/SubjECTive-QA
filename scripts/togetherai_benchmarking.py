import argparse
import re
import numpy as np
import pandas as pd
import time
from datetime import date
from datasets import load_dataset
from pathlib import Path
from together import Together
import logging

# Setting up basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Definitions for each feature to be evaluated
definition_map = {
    "RELEVANT": "The speaker has answered the question entirely and appropriately.",
    "SPECIFIC": "The speaker includes specific and technical details in the answer.",
    "CAUTIOUS": "The speaker answers using a more conservative, risk-averse approach.",
    "ASSERTIVE": "The speaker answers with certainty about the company's events.",
    "CLEAR": "The speaker is transparent in the answer and about the message to be conveyed.",
    "OPTIMISTIC": "The speaker answers with a positive tone regarding outcomes.",
}

# Function to parse command-line arguments for model configuration
def parse_arguments():
    """
    Runs inference using a specified language model (LLM) to classify a feature in a dataset and save results.

    Args:
        args (argparse.Namespace): Parsed arguments containing the following:
            - feature (str): The feature to evaluate from the dataset.
            - model (str): The name of the LLM model to use.
            - api_key (str): API key for authenticating the LLM request.
            - max_tokens (int): Maximum number of tokens to generate in the LLM response.
            - temperature (float): Temperature value for controlling the randomness of the LLM output.
            - top_p (float): Top-p sampling value for nucleus sampling.
            - frequency_penalty (float): Penalty for repeating tokens in LLM output.
            - seed (str): Seed value for loading the dataset with reproducibility.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - questions: List of questions processed from the dataset.
            - answers: Corresponding answers from the dataset.
            - llm_responses: Responses generated by the LLM model for the feature.
            - actual_labels: Actual labels for the feature from the dataset.
            - complete_responses: Full LLM response objects.

    The function processes each question in the dataset, sends it to the LLM model, and stores the responses
    and actual labels. Results are incrementally saved to a CSV file after each iteration to allow progress tracking
    and prevent data loss in case of errors.


    Notes:
        - Results are saved incrementally to a CSV file in the format: `results/{feature}/{model}/{feature}_{dd_mm_yyyy}.csv`.
        - The function includes delays between LLM requests to avoid hitting API rate limits.
        - Errors encountered during processing will be logged, and the function will retry with a longer delay.

    Raises:
        Exception: If an error occurs during the processing of a dataset row, it will be logged and the row will be retried after a delay.
    """
    parser = argparse.ArgumentParser(description="Run a LLM on TogetherAI")
    parser.add_argument("--model", type=str, help="Model to use", required=True)
    parser.add_argument("--feature", type=str, help="Feature to evaluate", required=True)
    parser.add_argument("--api_key", type=str, help="API key to use", required=True)
    parser.add_argument("--max_tokens", type=int, default=128, help="Max tokens to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature to use")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top-p to use")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k to use")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty to use")
    parser.add_argument("--seed", type=str, default='5768', help="Seed to use")
    return parser.parse_args()

# Function to extract a specific label from the text based on a regex pattern
def extract_label(text, label_regex):
    match = re.search(label_regex, text)
    return match.group(1) if match else "None"

# Main inference function to perform evaluations on the dataset using the LLM
def inference(args):
    """
    Runs inference using a specified language model (LLM) to classify a feature in a dataset and save results.

    Args:
        args (argparse.Namespace): Parsed arguments containing the following:
            - feature (str): The feature to evaluate from the dataset.
            - model (str): The name of the LLM model to use.
            - api_key (str): API key for authenticating the LLM request.
            - max_tokens (int): Maximum number of tokens to generate in the LLM response.
            - temperature (float): Temperature value for controlling the randomness of the LLM output.
            - top_p (float): Top-p sampling value for nucleus sampling.
            - frequency_penalty (float): Penalty for repeating tokens in LLM output.
            - seed (str): Seed value for loading the dataset with reproducibility.

    Returns:
        pandas.DataFrame: A DataFrame containing the following columns:
            - questions: List of questions processed from the dataset.
            - answers: Corresponding answers from the dataset.
            - llm_responses: Responses generated by the LLM model for the feature.
            - actual_labels: Actual labels for the feature from the dataset.
            - complete_responses: Full LLM response objects.

    The function processes each question in the dataset, sends it to the LLM model, and stores the responses
    and actual labels. Results are incrementally saved to a CSV file after each iteration to allow progress tracking
    and prevent data loss in case of errors.


    Notes:
        - Results are saved incrementally to a CSV file in the format: `results/{feature}/{model}/{feature}_{dd_mm_yyyy}.csv`.
        - The function includes delays between LLM requests to avoid hitting API rate limits.
        - Errors encountered during processing will be logged, and the function will retry with a longer delay.

    Raises:
        Exception: If an error occurs during the processing of a dataset row, it will be logged and the row will be retried after a delay.
    """
    client = Together()  # Initialize the Together client
    feature = args.feature.strip('“”"')  # Clean up the feature input
    dataset = load_dataset("gtfitechlab/SubjECTive-QA", split="test", seed=args.seed)

    # Lists to store various information (questions, answers, etc.)
    questions = []
    answers = []
    llm_responses = []
    feature_labels = []
    complete_responses = []

    # Define the path to save the results
    results_path = (
        Path.cwd()
        / "results"
        / feature
        / args.model
        / f"{feature}_{date.today().strftime('%d_%m_%Y')}.csv"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Iterate through the 'test' split of the dataset
    start_t = time.time()
    for i, row in enumerate(dataset["test"]):  # Access dataset using the dictionary key
        question = row["QUESTION"]
        answer = row["ANSWER"]
        actual_label = row[feature]
        
        # Append the extracted data to respective lists
        questions.append(question)
        answers.append(answer)
        feature_labels.append(actual_label)
        
        logger.info(f"Processing question {i + 1}/{len(dataset['test'])}: {question}")
        
        try:
            # Sending the question-answer pair to the model for evaluation
            model_response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are an expert sentence classifier."},
                    {"role": "user", "content": prompt(feature, definition_map[feature], question, answer)},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty
            )
            
            # Extract model response and append to list
            complete_responses.append(model_response)
            response_label = model_response.choices[0].message.content
            llm_responses.append(response_label)
            
            logger.info(f"Processed row {i + 1}/{len(dataset['test'])}: {response_label}")
            
            # Continually save progress after processing each row
            df = pd.DataFrame({
                "questions": questions,
                "answers": answers,
                "llm_responses": llm_responses,
                "actual_labels": feature_labels,
                "complete_responses": complete_responses,
            })
            df.to_csv(results_path, index=False)  # Save the DataFrame to CSV
            logger.info(f"Saved progress to {results_path}")

            # Adding a delay between API calls to avoid hitting rate limits
            time.sleep(7.0)
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            time.sleep(8.0)  # Additional delay in case of an error

    logger.info(f"Final results saved to {results_path}")
    return df

# Function to generate the prompt for the model based on the feature and definition
def prompt(feature, definition, question, answer):
    return f"""Given the following feature: {feature} and its corresponding definition: {definition}\n
              Rate the answer with:\n
              2: Positive demonstration of the feature in the answer.\n
              1: Neutral or no correlation.\n
              0: Negative correlation to the question on the feature.\n
              Provide the rating only. This is the question: {question}, and this is the answer: {answer}."""


# Main function to run the inference after parsing arguments
def main():
    args = parse_arguments()  # Parse command-line arguments
    feature = args.feature.strip('“”"')  # Clean the feature input

    # Ensure the feature is valid before starting the inference
    if feature in definition_map:
        start_t = time.time()
        df = inference(args) 
        time_taken = time.time() - start_t
        logger.info(f"Processing time: {time_taken} seconds")
    else:
        logger.error(f"Feature '{feature}' not found in feature definition map.")

if __name__ == "__main__":
    main()