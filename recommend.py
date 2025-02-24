import pandas as pd
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Loads the dataset
df = pd.read_csv("movies.csv")[['title', 'genre', 'storyline']].dropna()

#Cleans the input by converting to lowercase and removing punctuation and special characters.
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    return text

df['cleaned_storyline'] = df['storyline'].apply(clean_text)

#Converts the cleaned storylines into numerical vectors using TF-IDF.
vectorizer = TfidfVectorizer(stop_words='english')
storyline_vectors = vectorizer.fit_transform(df['cleaned_storyline'])

#Takes a user input, converts it into a vector, and finds the most similar movies.
def get_recommendations(user_input, top_n=5):
    cleaned_input = clean_text(user_input)  
    input_vector = vectorizer.transform([cleaned_input]) 
    similarity_scores = cosine_similarity(input_vector, storyline_vectors).flatten()  

    top_matches = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_matches][['title', 'genre']] #Ensures only the title and genre of the movies are returned.
    
    return recommendations

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        recommended_movies = get_recommendations(user_query)

        print("\nRecommended Movies:")
        for i, row in enumerate(recommended_movies.itertuples(index=False), start=1):
            print(f"{i}. {row.title} - {row.genre}")
    else:
        print("Use: python recommend.py 'Enter description here'")