import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# Download the stopwords corpus if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def process_files():
    files = ['articles1.csv', 'articles2.csv', 'articles3.csv']
    chunksize = 10000
    all_chunks = []
    
    for file in files:
        print(f"Processing {file}...")
        try:
            # Read CSV in chunks
            chunk_iterator = pd.read_csv(file, chunksize=chunksize)
            
            for chunk in tqdm(chunk_iterator, desc=f"Chunks in {file}"):
                # Drop rows with NaN in 'content'
                if 'content' in chunk.columns:
                    chunk = chunk.dropna(subset=['content'])
                    
                    # Apply text cleaning
                    chunk['content'] = chunk['content'].apply(clean_text)
                    
                    # Remove completely empty texts after cleaning
                    chunk = chunk[chunk['content'].str.strip() != ""]
                    
                    all_chunks.append(chunk)
                else:
                    print(f"'content' column not found in {file}! Skipping chunk.")
        except FileNotFoundError:
            print(f"{file} not found. Skipping.")
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    if all_chunks:
        print("Concatenating all processed chunks...")
        final_df = pd.concat(all_chunks, ignore_index=True)
        
        print("Saving to cleaned_news.parquet...")
        final_df.to_parquet('cleaned_news.parquet', engine='pyarrow')
        print("Save complete!")
    else:
        print("No valid data processed. Output file was not created.")

if __name__ == "__main__":
    process_files()
