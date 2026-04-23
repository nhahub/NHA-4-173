# Data Preprocessing Documentation for LLM Fine-Tuning

This document outlines the steps taken to clean and preprocess the news article files (`articles1.csv`, `articles2.csv`, `articles3.csv`) in preparation for Large Language Model (LLM) Fine-Tuning. It explains the technical reasoning behind each step and details the techniques intentionally excluded.

---

## 1. Executed Steps (What we did)

We built the `preprocess.py` script to perform the following operations:

### A. Efficient Memory Management
* **What we did:** Used the `chunksize=10000` parameter in the Pandas library.
* **Reason:** The dataset files are massive. Reading them completely into memory at once would cause a RAM crash. Dividing data into chunks ensures stable and fast processing.

### B. Handling Missing Values (NaN)
* **What we did:** Used `dropna` to remove any rows lacking text in the `content` column.
* **Reason:** Feeding null/empty values into the model during training will trigger coding errors and immediately crash the training process.

### C. Text Cleaning
A comprehensive cleaning function was applied to each article, including:
1. **Lowercasing:** Converting all text characters to lowercase to prevent the model from treating "Apple" and "apple" as entirely distinct words.
2. **Removing URLs:** Utilizing Regex to strip website links (http/https/www), as they carry no useful semantic value for text generation.
3. **Removing HTML Tags:** Cleaning the text of residual web page code snippets (like `<div>` or `<p>`).
4. **Removing Punctuation:** Eliminating commas, periods, and symbols to unify words and shrink the vocabulary size. *(Note: For advanced LLM text generation tasks where grammatically perfect sentence structure is critical, it is sometimes preferred to keep punctuation).*
5. **Removing Stop Words:** Using `NLTK`, we deleted frequent, low-meaning words (like the, is, at, in) to minimize distraction for the model and reduce dataset size.

### D. Final Format Selection (Parquet vs CSV)
* **What we did:** Saved the concatenated, cleaned data into a single `cleaned_news.parquet` file.
* **Reason:** The Parquet format uses columnar data storage and advanced file compression. It drastically saves disk space, preserves data schemas, and loads into AI data pipelines exponentially faster than a standard CSV.

---

## 2. Skipped Techniques (What we didn't do, and WHY)

In traditional Natural Language Processing (NLP), there are fundamental steps that we deliberately excluded here. Here is why:

### 1️⃣ Why No Stemming or Lemmatization?
* **Concept:** Reversing words back to their base/root form (e.g., turning "went", "going", "goes" into the root "go").
* **Reason for Exclusion:** Our goal is to train a **Generative LLM**. In order for the model to synthesize natural, human-like sentences, it *must* learn grammar and tenses. Reducing words to their roots destroys the temporal context of the text, forcing the model to generate broken or grammatically incorrect sentences (e.g., "He go yesterday" instead of "He went yesterday").

### 2️⃣ Why No Tokenization?
* **Concept:** Splitting text into smaller sub-word units (Tokens) and converting them into numerical IDs the model can understand.
* **Reason for Exclusion:** The tokenization step is **strictly tied to the specific Model** you intend to train.
  * If you use `LLaMA`, it utilizes a specific Tokenizer (BPE/SentencePiece).
  * If you use `GPT`, it utilizes a completely different Tokenizer (Tiktoken).
* **The Correct Approach:** In LLM pipelines, we save the data as purely clean plain text (in a Parquet file, as done here). Tokenization is executed "on-the-fly" during the actual training phase utilizing the exact Tokenizer for the selected model via libraries like Hugging Face `transformers`.

---

## 3. Exploratory Data Analysis (EDA)

We also built smart, memory-efficient EDA scripts:

1. **`eda_before.py` (Pre-cleaning):** 
   Reads an equal, manageable sample (approx. 50,000 rows limit) directly from the raw files without overloading the RAM. It generates: a histogram of text lengths, the most frequent top 10 authors, and a Word Cloud per file, saving the plots independently for visual comparison.
2. **`eda_after.py` (Post-cleaning):** 
   Loads the cleaned `Parquet` file and extracts a random sample of 50,000 rows to generate the identical plots. This enables us to visualize the impact of our cleaning function and observe how the Word Cloud transforms after removing noisy data.

---

## 4. Next Steps

### Model-Specific Tokenization
The immediate next step in our ML pipeline is **Tokenization**. Since tokenization logic is highly model-dependent, we will:
1. Select the target LLM for fine-tuning (e.g., LLaMA, Mistral, or a BERT variant).
2. Load the specific Tokenizer designed for that model (e.g., using Hugging Face `transformers`).
3. Apply the tokenizer to our `cleaned_news.parquet` text data (often on-the-fly using `Dataset.map()`), converting the words into numerical IDs ready for the model's neural network.