# Cellula_Week_2
##Toxic Content Classification App (Streamlit)

This project was developed as part of the Cellula Technologies Internship – Week 2.

It is a Streamlit application that detects toxic content from:

User-entered text

Uploaded images (through generated captions)

The system combines a fine-tuned transformer-based toxicity classifier with an image captioning model.

Project Features

Text toxicity classification

Image toxicity detection through caption generation

Fine-tuned DistilBERT model using LoRA adapters (PEFT)

BLIP image captioning implemented in a separate module

Automatic CSV logging of all user inputs and predictions

Option to view the stored database inside the app

Implemented using only Python files (.py), no notebooks

Models Used
Toxicity Classifier

Base Model: distilbert-base-uncased

Fine-tuning Method: LoRA (Low-Rank Adaptation)

Dataset: Hugging Face civil_comments

Output Classes:

Safe

Toxic

LoRA adapters were applied to DistilBERT attention modules:

q_lin

v_lin

Adapter weights and tokenizer are stored in:

toxic_lora_model/

Image Captioning

Model: BLIP (Salesforce)

Implemented in: imagecaption.py

Uploaded images are first converted into captions, then classified using the toxic text classifier.
