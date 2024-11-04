# SubjECTive-QA: Measuring Subjectivity in Earnings Call Transcripts’ QA Through Six-Dimensional Feature Analysis. 

**Authors**: Huzaifa Pardawala, Siddhant Sukhani, Agam Shah, Veer Kejriwal, Rohan Bhasin, Abhishek Pillai, Dhruv Adha, Tarun Mandapati, Andrew DiBiasio, Sudheer Chava

---



[Access the SubjECTiveQA Paper here](https://arxiv.org/abs/2410.20651)

### Dataset Availability

Access the dataset on [Hugging Face](https://huggingface.co/datasets/gtfintechlab/SubjECTive-QA)
<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo" width="25"/>

The SubjECTive-QA dataset is available on Hugging Face. You can load the dataset using the following code:

```python
from datasets import load_dataset

dataset = load_dataset("gtfintechlab/subjectiveqa", seed={SEED})
```
### Available seeds:

- 5768
- 78516
- 944601
---


<div align="center">
  <img src="https://github.com/user-attachments/assets/ca68a7cf-e6d5-4184-8c0e-7895a94d7691" alt="misinformation_example" width="600"/>
</div>


## Abstract
Fact-checking is extensively studied in the context of misinformation and disinformation, addressing objective inaccuracies. However, a softer form of misinformation involves responses that are factually correct but lack features such as clarity and relevance. This challenge is prevalent in formal Question-Answer (QA) settings, like press conferences in finance, politics, and sports, where subjective answers can obscure transparency. Despite this, there is a lack of manually annotated datasets for subjective features across multiple dimensions. To address this gap, we introduce **SubjECTive-QA**, a dataset manually annotated on Earnings Call Transcripts (ECTs) by nine annotators. The dataset includes 2,747 annotated long-form QA pairs across six features: Assertive, Cautious, Optimistic, Specific, Clear, and Relevant.

Benchmarking on our dataset reveals that the Pre-trained Language Model (PLM) RoBERTa-base has similar weighted F1 scores to Llama-3-70b-Chat on features with lower subjectivity, like Relevant and Clear, with a mean difference of 2.17% in their weighted F1 scores. On features with higher subjectivity, like Specific and Assertive, RoBERTa-base significantly outperforms Llama-3-70b-Chat, with a mean difference of 10.01% in weighted F1 scores. Furthermore, generalizing SubjECTive-QA to White House Press Briefings and Gaggles demonstrates broader applicability, with an average weighted F1 score of 65.97%.

**SubjECTive-QA** is available under the CC BY 4.0 license.

<div align="center">
  <img src="https://github.com/user-attachments/assets/b17c9b63-a838-4529-8c07-08d3b9bdf3e8" alt="methodology" width="800"/>
</div>


---


### Model Availability

The SubjECTive-QA models are also available on Hugging Face <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face Logo" width="25"/>. You can perform inference with the models using the following code:

```python
# Imports
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the model and tokenizer for a particular feature
model = AutoModelForSequenceClassification.from_pretrained("gtfintechlab/SubjECTiveQA-{FEATURE}")
tokenizer = AutoTokenizer.from_pretrained("gtfintechlab/SubjECTiveQA-{FEATURE}")

# Prepare your input text
text = ""

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Perform the inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the logits (raw model outputs)
logits = outputs.logits

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Get the predicted class index
predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

# Map the class index to the actual label
labels = model.config.id2label
predicted_label = labels[predicted_class_idx]

print(f"Predicted class index: {predicted_class_idx}")
print(f"Predicted label: {predicted_label}")
```

1. Access the fine-tuned model with the best hyperparameters for `CLEAR` on [Hugging Face](https://huggingface.co/gtfintechlab/SubjECTiveQA-CLEAR).


2. Access the fine-tuned model with the best hyperparameters for `RELEVANT` on [Hugging Face](https://huggingface.co/gtfintechlab/SubjECTiveQA-RELEVANT).

3. Access the fine-tuned model with the best hyperparameters for `CAUTIOUS` on [Hugging Face](https://huggingface.co/gtfintechlab/SubjECTiveQA-CAUTIOUS).

4. Access the fine-tuned model with the best hyperparameters for `ASSERTIVE` on [Hugging Face](https://huggingface.co/gtfintechlab/SubjECTiveQA-ASSERTIVE).

5. Access the fine-tuned model with the best hyperparameters for `OPTIMISTIC` on [Hugging Face](https://huggingface.co/gtfintechlab/SubjECTiveQA-OPTIMISTIC).

6. Access the fine-tuned model with the best hyperparameters for `SPECIFIC` on [Hugging Face](https://huggingface.co/gtfintechlab/SubjECTiveQA-SPECIFIC).

---

## Running the Benchmarking Scripts

To run the benchmarking on different models, use the following commands:

### OpenAI Benchmarking
```bash
python3 openai_benchmarking.py --model "" --feature "" --api_key "" --max_tokens "" --temperature "" --top_p "" --frequency_penalty ""
```

### Together.AI Benchmarking
```bash
python3 togetherai_benchmarking.py --model "" --feature "" --api_key "" --max_tokens "" --temperature "" --top_p "" --top_k "" --repetition_penalty ""
```
### Pre-Trained Language Models Benchmarking
```bash
python3 plm_benchmarking.py
```

---

### License

The SubjECTive-QA dataset is licensed under the Creative Commons Attribution 4.0 International License.

### Contact
For any questions or concerns, feel free to reach out to huzaifahp7@gmail.com.

### Citation: If you use our open-source dataset or refer to our results, please cite our paper:


This work has been accepted at the **38th Conference on Neural Information Processing Systems (NeurIPS 2024)**, Datasets and Benchmarks Track.

<div align="left">
  <img src="https://github.com/user-attachments/assets/1fbd87b6-3fb3-43ec-8e61-855cbf868977" alt="90" width="100"/>
</div>


```bash
@misc{pardawala2024subjectiveqameasuringsubjectivityearnings,
      title={SubjECTive-QA: Measuring Subjectivity in Earnings Call Transcripts' QA Through Six-Dimensional Feature Analysis}, 
      author={Huzaifa Pardawala and Siddhant Sukhani and Agam Shah and Veer Kejriwal and Abhishek Pillai and Rohan Bhasin and Andrew DiBiasio and Tarun Mandapati and Dhruv Adha and Sudheer Chava},
      year={2024},
      eprint={2410.20651},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.20651}, 
}
```
