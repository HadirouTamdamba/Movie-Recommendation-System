import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib

# Load datas
movies = pd.read_csv("data/movies.csv", sep=",", engine="python", header=None, names=["movie_id", "title", "genres"])
ratings = pd.read_csv("data/ratings.csv", sep=",", engine="python", header=None, names=["user_id", "movie_id", "rating", "timestamp"])

# Preprocessing
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Comparison of models
models = {
    "SVD": SVD(),
    "KNNBasic": KNNBasic(),
    "NMF": NMF()
}

results = {}
for name, model in models.items():
    model.fit(trainset)
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    results[name] = rmse
    print(f"{name} - RMSE: {rmse}")

# Saving the best model
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
best_model

joblib.dump(best_model, "models/best_movie_recommendation_model.pkl")