import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def run_eda_after():
    print("Loading cleaned data...")
    # Read the cleaned parquet data
    df = pd.read_parquet('cleaned_news.parquet')
    
    # Sample 50,000 rows for faster processing
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        
    df = df.dropna(subset=['content', 'author'])
    
    print("Generating visualizations for CLEANED data...")
    # Calculate word count after cleaning
    df['word_count'] = df['content'].astype(str).apply(lambda x: len(x.split()))

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('EDA After Cleaning (Processed Data)', fontsize=20, fontweight='bold', color='darkblue')

    # Visualizations
    # 1. Histogram
    ax1 = fig.add_subplot(2, 2, 1)
    sns.histplot(df['word_count'], bins=50, color='skyblue', ax=ax1)
    ax1.set_title('Distribution of Article Word Counts (Cleaned)', fontsize=14)
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Frequency')
    ax1.set_xlim(0, df['word_count'].quantile(0.95))

    # 2. Bar Chart
    ax2 = fig.add_subplot(2, 2, 2)
    top_authors = df['author'].value_counts().head(10)
    sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax2, hue=top_authors.index, palette='viridis', legend=False)
    ax2.set_title('Top 10 Authors/Sources', fontsize=14)
    ax2.set_xlabel('Number of Articles')

    # 3. Word Cloud
    ax3 = fig.add_subplot(2, 1, 2)
    all_text = " ".join(df['content'].astype(str).tolist())
    wordcloud = WordCloud(width=1600, height=400, background_color='white', max_words=150, colormap='magma').generate(all_text)
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis('off')
    ax3.set_title('Most Frequent Words After Cleaning', fontsize=14)

    plt.tight_layout(pad=3.0)
    
    # Save the plot and show
    plt.savefig('eda_after_cleaning.png')
    print("Saved plot to 'eda_after_cleaning.png'")
    plt.show()

if __name__ == "__main__":
    run_eda_after()
