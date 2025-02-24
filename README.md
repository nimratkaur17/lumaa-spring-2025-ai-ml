Movie Recommendation System

This is a simple content-based recommendation system that suggests movies based on a short description provided by the user. It uses TF-IDF vectorization and cosine similarity to find the most relevant movies.

# Dataset
The dataset used is the IMDB Top 250 Movies dataset from Kaggle. It contains various columns, including movie titles, genres, and storylines. The dataset will be included in the forked repository.

# Setup
Ensure Python version 3.12.7 is installed.
Check using: python --version
Install required libraries using: pip install pandas scikit-learn

# Running
To run the code, use: python recommend.py "Enter user description"

# Results
python recommend.py "I like romance movies with some suspenseful thriller aspects." 
Output:

Recommended Movies:
1. It's a Wonderful Life - Drama,Family,Fantasy
2. The Apartment - Comedy,Drama,Romance
3. Gone with the Wind - Drama,Romance,War
4. Dances with Wolves - Adventure,Drama,Western
5. Reservoir Dogs - Crime,Thriller
