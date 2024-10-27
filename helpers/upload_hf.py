import os
import sys
from pathlib import Path
from huggingface_hub import login
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict
import logging

SRC_DIRECTORY = Path().cwd().resolve()
# DATA_DIRECTORY = Path().cwd().resolve().parent + "/data"
DATA_DIRECTORY = "../final_dataset"


if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

HF_TOKEN = os.getenv("HF_TOKEN")
HF_ORGANIZATION = "subjectiveqa"
DATASET = "subjectiveqa"
login(HF_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def huggify_data_subjectiveqa(push_to_hub=False, TASK=None, SEED=None, SPLITS=['train','test','val']):
    try:
        directory_path = DATA_DIRECTORY
        logger.debug(f"Directory path: {directory_path}")

        hf_dataset = DatasetDict()
        
        df = pd.read_csv(f"{directory_path}/final_dataset.csv")

        input_columns = ["COMPANYNAME","QUARTER","YEAR","ASKER","RESPONDER","QUESTION","ANSWER"]
        output_columns = ["CLEAR", "ASSERTIVE", "CAUTIOUS", "OPTIMISTIC", "SPECIFIC", "RELEVANT"]
        
        train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=SEED)
        test_df, val_df = train_test_split(test_val_df, test_size=0.3, random_state=SEED)
        
        
        hf_dataset['train'] = Dataset.from_pandas(train_df[input_columns + output_columns])

        # Add test split
        hf_dataset['test'] = Dataset.from_pandas(test_df[input_columns + output_columns])
        
        hf_dataset['val'] = Dataset.from_pandas(val_df[input_columns + output_columns])

        # Push to HF Hub
        if push_to_hub:
            hf_dataset.push_to_hub(
                f"{HF_ORGANIZATION}/{DATASET}",
                config_name= str(SEED),
                private=True,
                token=HF_TOKEN,
            )

            # TODO: push the dataset dict object not the datasets individually

        logger.info("Finished processing SubjECTive dataset seed : {SEED}")
        return hf_dataset

    except Exception as e:
        logger.error(f"Error processing SubjECTive dataset: {str(e)}")
        raise e


if __name__ == "__main__":
    SPLITS = ['train', 'test', 'val']

    TASK = "SubjECTive-QA"

    SEEDS = (5768, 78516, 944601)
    
    

    for SEED in list(reversed(SEEDS)):
        huggify_data_subjectiveqa(push_to_hub=True, TASK=TASK, SEED=SEED, SPLITS=SPLITS)

