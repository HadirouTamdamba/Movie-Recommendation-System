# Movie Recommendation System

Link to database : https://grouplens.org/datasets/movielens/ 

## Problem Statement
The goal is to recommend movies to users based on their preferences. This project is useful for streaming platforms like Netflix or Disney+.

## Dataset Description
The **MovieLens** dataset contains:
- **ratings.dat**: 32 million user ratings.
- **movies.dat**: Movie information (title, genres).

## Exploratory Data Analysis (EDA)
- Distribution of ratings.
- Number of movies by genre.

## Modeling
Comparison of several models:
- **SVD** (Singular Value Decomposition)
- **KNNBasic** (k-Nearest Neighbors)
- **NMF** (Non-Negative Matrix Factorization)

## Evaluation
The best model is selected based on **RMSE** (Root Mean Squared Error).

## Deployment
A Flask API is created to recommend movies in real-time.

## Conclusion
TThis project demonstrates how AI can personalize recommendations for users.



