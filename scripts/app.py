from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Upload the best model
model = joblib.load("models/best_movie_recommendation_model.pkl")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    user_id = data["user_id"]
    n_recommendations = data.get("n_recommendations", 5)
    
    # Generate recommendations
    all_movies = pd.read_csv("data/movies.csv", sep=",", engine="python", header=None, names=["movie_id", "title", "genres"])["movie_id"].unique()
    user_movies = pd.read_csv("data/ratings.csv", sep=",", engine="python", header=None, names=["user_id", "movie_id", "rating", "timestamp"])
    user_movies = user_movies[user_movies["user_id"] == user_id]["movie_id"].unique()
    movies_to_predict = np.setdiff1d(all_movies, user_movies) 
    
    predictions = []
    for movie_id in movies_to_predict:
        predictions.append((movie_id, model.predict(user_id, movie_id).est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:n_recommendations]
    
    return jsonify({"recommendations": top_movies})

if __name__ == "__main__":
    app.run(debug=True)