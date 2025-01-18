import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
movies = pd.read_csv("data/movies.csv", sep=",", engine="python", header=None, names=["movie_id", "title", "genres"])
ratings = pd.read_csv("data/ratings.csv", sep=",", engine="python", header=None, names=["user_id", "movie_id", "rating", "timestamp"])



# Distribution of ratings
plt.figure(figsize=(10, 6))
sns.histplot(ratings["rating"], bins=10, kde=True)
plt.title("Distribution of ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Number of movies by genre
genres_count = movies["genres"].str.get_dummies("|").sum()
genres_count.sort_values(ascending=False).plot(kind="bar", figsize=(12, 6))
plt.title("Number of movies by genre")
plt.xlabel("Genre")
plt.ylabel("Number of movies")
plt.show() 