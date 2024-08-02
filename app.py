# Standard library imports
import json
import os

from itertools import chain, tee
from typing import List, Tuple

# Third-party imports
import numpy as np
import openai
import pandas as pd
import spacy
import streamlit as st
import torch

from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertModel, BertTokenizer

st.title('Hercules.AI Test Task')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

with st.spinner('Loading the Spacy model...'):   
    nlp = spacy.load("en_core_web_sm")

with st.spinner('Loading the T5 model...'): 
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
with st.spinner('Loading BERT model...'):
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

def process_text(text: str) -> str:
    """
    Process the given text by lemmatizing tokens and removing stop words and punctuation.

    Args:
        text (str): The input text to process.

    Returns:
        str: The processed text with lemmatized tokens and stop words and punctuation removed.
    """
    doc = nlp(text)
    terms = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return terms

def generate_ngrams(terms: List[str], n: int = 2) -> List[str]:
    """
    Generate n-grams from a list of terms.

    Args:
        terms (List[str]): The list of terms to generate n-grams from.
        n (int, optional): The maximum length of the n-grams. Defaults to 2.

    Returns:
        List[str]: A list of n-grams generated from the input terms.
    """
    def ngrams(iterable: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Generate n-grams from an iterable.

        Args:
            iterable (List[str]): The iterable to generate n-grams from.
            n (int): The length of the n-grams.

        Returns:
            List[Tuple[str, ...]]: A list of n-grams.
        """
        iters = tee(iterable, n)
        for i, it in enumerate(iters):
            for _ in range(i):
                next(it, None)
        return list(zip(*iters))
    
    n_grams = list(chain.from_iterable(ngrams(terms, i) for i in range(1, n + 1)))
    return [" ".join(gram) for gram in n_grams]
    

def embed_text(text: str, model: BertModel, tokenizer: BertTokenizer) -> np.ndarray:
    """
    Embed the given text using the specified BERT model and tokenizer.

    Args:
        text (str): The input text to embed.
        model (BertModel): The BERT model used for generating embeddings.
        tokenizer (BertTokenizer): The BERT tokenizer used for tokenizing the input text.

    Returns:
        np.ndarray: The text embedding as a numpy array.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def extract_terms_with_openai(contract_text: str) -> str:
    """
    Extract key terms and constraints from a contract text using the OpenAI API.

    Args:
        contract_text (str): The contract text to process.

    Returns:
        str: The extracted key terms and constraints.
    """
    openai.api_key = st.text_input("Enter your OpenAI API key:", type="password", value='hidden!')
    
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=f"Identify terms in {contract_text}",
        max_tokens=7,
        temperature=0
    )
    return response['choices'][0]['message']['content'].strip()

def extract_terms_with_t5(contract_text: str) -> str:
    """
    Extract key terms and constraints from a contract text using the T5 model.

    Args:
        contract_text (str): The contract text to process.

    Returns:
        str: The extracted key terms and constraints.
    """
    input_text = (
        f"Extract key terms and constraints from the following contract text section by section:\n\n"
        f"Contract: {contract_text}\n\n"
        f"Please list the key terms and constraints."
    )

    inputs = tokenizer.encode_plus(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
        add_special_tokens=True
    )
    
    outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)
    terms = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return terms

def analyze_compliance(tasks: List[str], anchor_points: List[np.ndarray], threshold: float = 0.5) -> List[Tuple[str, bool]]:
    """
    Analyze the compliance of task descriptions with the extracted contract terms.

    Args:
        tasks (List[str]): The list of task descriptions.
        anchor_points (List[np.ndarray]): The list of embedded contract terms.
        threshold (float, optional): The similarity threshold for compliance. Defaults to 0.5.

    Returns:
        List[Tuple[str, bool]]: A list of tuples containing task descriptions and their compliance status.
    """
    compliance_report = []
    for task in tasks:
        task_embedding = embed_text(task, bert_model, bert_tokenizer)
        similarities = cosine_similarity([task_embedding], anchor_points)[0]
        compliance = any(similarity > threshold for similarity in similarities)
        compliance_report.append((task, compliance))
    return compliance_report

# File uploader for TXT files
uploaded_txt = st.file_uploader("Choose a .txt file with the contract", type=["txt"], key="txt_uploader")

# File uploader for CSV files
uploaded_csv = st.file_uploader("Choose a .csv file with the task description", type=["csv"], key="csv_uploader")

if uploaded_txt:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_txt.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_txt.getbuffer())
    st.success(f'File "{uploaded_txt.name}" uploaded successfully!')

    with open(file_path, 'r') as f:
        contract_text = f.read()

    model_choice = "T5" # hardcoded T5 to make the solution more stable and effective in the context of the API key privacy issues
    if model_choice == "T5":
        terms = extract_terms_with_t5(contract_text)
    else:
        terms = extract_terms_with_openai(contract_text)

    processed_terms = process_text(terms)
    ngrams = generate_ngrams(processed_terms)
    term_embeddings = [embed_text(term, bert_model, bert_tokenizer) for term in processed_terms]
    
    st.subheader("Extracted Terms and Conditions:")
    st.text_area("Key Terms:", value="\n".join(ngrams), height=300)
    
    terms_json = json.dumps(ngrams, indent=2)
    st.download_button(
        label="Download JSON",
        data=terms_json,
        file_name="extracted_terms.json",
        mime="application/json"
    )

if uploaded_csv and uploaded_txt:
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_csv.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_csv.getbuffer())
    st.success(f'File "{uploaded_csv.name}" uploaded successfully!')
    df = pd.read_csv(file_path)
    compliance_report = analyze_compliance(df['Task Description'].tolist(), term_embeddings)
    df['compliance'] = ["Compliant" if compliance else "Non-Compliant" for _, compliance in compliance_report]
    st.subheader("Compliance Report")
    st.write(df)
elif uploaded_csv and not uploaded_txt:
    st.error("Please upload a TXT file with contract terms before processing the CSV file.")

