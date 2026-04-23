import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def run_eda_before():
    print("Loading raw data and generating visualizations for each file...")
    
    files = ['articles1.csv', 'articles2.csv', 'articles3.csv']
    
    for file in files:
        try:
            print(f"\nProcessing {file}...")
            # Read a sample from each file to avoid excessive memory usage (e.g., 50k rows)
            df = pd.read_csv(file, nrows=50000)
            
            # Identify the correct text column
            text_col = 'content' if 'content' in df.columns else 'article'
            
            # Ensure there are no missing values in the required columns
            df = df.dropna(subset=[text_col, 'author'])
            
            if df.empty:
                print(f"Skipping {file} due to lack of valid data.")
                continue
                
            # Calculate word count before cleaning
            df['word_count'] = df[text_col].astype(str).apply(lambda x: len(x.split()))

            # Create a figure for this specific file
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle(f'EDA Before Cleaning - {file}', fontsize=20, fontweight='bold')

            # 1. Histogram
            ax1 = fig.add_subplot(2, 2, 1)
            sns.histplot(df['word_count'], bins=50, color='salmon', ax=ax1)
            ax1.set_title(f'Distribution of Word Counts ({file})', fontsize=14)
            ax1.set_xlabel('Word Count')
            ax1.set_ylabel('Frequency')
            ax1.set_xlim(0, df['word_count'].quantile(0.95))

            # 2. Bar Chart
            ax2 = fig.add_subplot(2, 2, 2)
            top_authors = df['author'].value_counts().head(10)
            sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax2, hue=top_authors.index, palette='rocket', legend=False)
            ax2.set_title(f'Top 10 Authors/Sources ({file})', fontsize=14)
            ax2.set_xlabel('Number of Articles')

            # 3. Word Cloud
            ax3 = fig.add_subplot(2, 1, 2)
            all_text = " ".join(df[text_col].astype(str).tolist())
            wordcloud = WordCloud(width=1600, height=400, background_color='white', max_words=150, colormap='inferno').generate(all_text)
            ax3.imshow(wordcloud, interpolation='bilinear')
            ax3.axis('off')
            ax3.set_title(f'Most Frequent Words ({file})', fontsize=14)

            plt.tight_layout(pad=3.0)
            
            # Save the plot with the file name
            save_name = f'eda_before_{file.replace(".csv", "")}.png'
            plt.savefig(save_name)
            print(f"Saved plot to '{save_name}'")
            
        except FileNotFoundError:
            print(f"Warning: {file} not found.")
            
    # Display all generated plots
    print("\nDisplaying all generated plots...")
    plt.show()

if __name__ == "__main__":
    run_eda_before()
