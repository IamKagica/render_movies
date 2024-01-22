### Uses code from the following:
### Title: Project 4 App (Python): What we have tried

import pandas as pd
import numpy as np
import os
import requests

# Define the URL for movie data
myurl = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"

# Fetch the data from the URL
response = requests.get(myurl)

# Split the data into lines and then split each line using "::"
movie_lines = response.text.split('\n')
movie_data = [line.split("::") for line in movie_lines if line]

# Create a DataFrame from the movie data
movies = pd.DataFrame(movie_data, columns=['movie_id', 'title', 'genres'])
movies['movie_id'] = movies['movie_id'].astype(int)

genres = list(
    sorted(set([genre for genres in movies.genres.unique() for genre in genres.split("|")]))
)

def get_displayed_movies():
    return movies.head(100)

def dict_to_df(new_user_dict):
    
    # Load the CSV file containing R into a DataFrame
    R = pd.read_csv("Rmat.csv")
    
    # Copy a DataFrame
    unew_ratings = R.iloc[0, :]
    # Replace all values with NaN using numpy.nan
    unew_ratings = unew_ratings * np.nan
        
    # Loop over all keys in the dictionary
    for key in new_user_dict:
        i_index = int(key) - 1
        unew_ratings.iloc[i_index] = new_user_dict[key]
    
    return unew_ratings
    
def get_recommended_movies(new_user_ratings):
    
    # Convert the dictionary to a DataFrame if needed
    # Check if the variable is a dictionary
    if isinstance(new_user_ratings, dict):
        newuser = dict_to_df(new_user_ratings)
    else:
        newuser = new_user_ratings
        
    # Using IBCF function
    
    # Loading similarity matrix
    # S = pd.read_csv('Symmetry_top30.csv')
    S_numrows = pd.read_csv('Symmetry_top30.csv', usecols=[0])
    
    # Load the CSV file containing R into a DataFrame
    R = pd.read_csv("Rmat.csv")
    
    num_movies = np.shape(S_numrows)[0]
    # num_movies = np.shape(S)[0]
    # num_movies = (S)[0]
    list_scores = np.zeros(num_movies)
    
    # Calculating predictions
    for l in range(num_movies):
        
        jump = 50
        # Reading fifty lines at a time
        if (l % jump == 0):
            subtract = l
            S_partial = pd.read_csv("Symmetry_top30.csv", skiprows=l, nrows=jump)
            print(l)
            
        # Reading one line at a time
        # Sl = pd.read_csv("Symmetry_top30.csv", skiprows=l, nrows=1)
        
        # Sl = S.iloc[l, :]
        
        Sl = S_partial.iloc[l - subtract, :]
        Sl_w = Sl.to_numpy() * newuser.to_numpy()
        numerator = np.nansum(Sl_w)
        
        Sl_1 = Sl.to_numpy() * (newuser.to_numpy() / newuser.to_numpy())
        denominator = np.nansum(Sl_1)
        
        if (np.isnan(newuser.iloc[l]) == False):
            # Will set movies that have already been seen as lowest priority.
            list_scores[l] = -1
        elif (np.isnan(denominator) == True or denominator == 0):
            # If there aren't enough predictions, will set to zero. 
            list_scores[l] = 0
        else:
            score_pred = (numerator / denominator)
            list_scores[l] = score_pred
                
    # Get indices of the top 10 largest values
    top_indices = np.argsort(list_scores)[-10:][::-1]
    
    unew_movie_nums = []
    
    print(top_indices)
    
    for i in range(len(top_indices)):
                
        # If you have the index of a column, you can get its name
        column_index = top_indices[i]  # Getting the desired index
        column_name_at_index = R.columns[column_index]
        unew_movie_nums.append(column_name_at_index) 
    
    # Top 10 highest values 
    N = 10
        
    # Get the titles of the top ten scoring movies
    # Making sure that we keep the same structure as before
    IBCF_dict = pd.DataFrame(columns=['movie_id', 'title', 'genres'])
    for i in range(N):
        for j in range(len(movies)):
            if (unew_movie_nums[i] == ('m' + str(movies.iloc[j].loc['movie_id']))):
                IBCF_dict.loc[i] = movies.iloc[j, :]
                    
    return IBCF_dict

def genre_movies(genre: str):
    
    # Using function from notebook
    liked_genre = genre
        
    # Create an empty DataFrame
    genre_subset = pd.DataFrame(columns=['movie_id', 'title', 'genres'])
        
    # Getting IDs of all movies that are part of the genre
    movie_num = []
    for i in range(len(movies)):
        if (liked_genre in movies.iloc[i].loc['genres']):
            new_row = movies.iloc[i]
            genre_subset = pd.concat([genre_subset, new_row], ignore_index=True)
            movie_num.append('m' + str(movies.iloc[i].loc['movie_id']))
        
    # Load the CSV file containing R into a DataFrame
    R = pd.read_csv("Rmat.csv")
    
    # Get scores for all movies in the genre
    movie_rec_score = []
    for i in range(len(movie_num)):
        this_movie_num = movie_num[i]
        # Subtraction by 3 to get center
        if this_movie_num not in R.columns:
            # No reviews means zero score
            this_movie_score = 0
        else:
            # Subtraction by 3 to get center
            this_movie_ratings = R[this_movie_num] - 3
            this_movie_score = np.nansum(this_movie_ratings)
        movie_rec_score.append(this_movie_score)

    # Get indices of the top 10 highest values
    N = 10
    indices_of_top_values = np.argsort(movie_rec_score)[-N:][::-1]
       
    # Get the top ten scoring movieIDs
    popular_genre_IDs = []
    for i in range(N):
        thisID = movie_num[indices_of_top_values[i]]
        popular_genre_IDs.append(thisID)
        
    # Get the titles of the top ten scoring movies
    # Making sure that we keep the same structure as before
    popular_genre_movies = pd.DataFrame(columns=['movie_id', 'title', 'genres'])
    for i in range(N):
        for j in range(len(movies)):
            if (popular_genre_IDs[i] == ('m' + str(movies.iloc[j].loc['movie_id']))):
                popular_genre_movies.loc[i] = movies.iloc[j, :]
        
    return popular_genre_movies

def store_movies(genres):
    
    # Save your top movie recommendations 
    # for each genre (e.g., in a table) 
    # to avoid recomputing them each time.
    
    # Top 10 highest values 
    N = 10
    
    genres_dict = movies.head(N)
    genres_dict = genres_dict.iloc[0:0]  # Empty the new DataFrame
        
    for genre_type in genres:
        genres_dict = pd.concat([genres_dict, genre_movies(genre_type)], ignore_index=True)
        
    genres_dict["recommended_type"] = "Any"
    for i in range(len(genres)):
        genre_type_i = genres[i]
        genres_dict.loc[(i*N):((i+1)*N), "recommended_type"] = genre_type_i
        
    return genres_dict
    
#####

# Used in get_popular_movies
genre_movies_path = "genre_movies.csv"
if os.path.exists(genre_movies_path):
    genres_dict = pd.read_csv(genre_movies_path)
else:
    genres_dict = store_movies(genres)
    # Saving and Loading Dataframe
    genres_dict.to_csv(genre_movies_path, index=False)
    genres_dict = pd.read_csv(genre_movies_path)
#####

def get_popular_movies(genre: str):
    
    popular_movies = genres_dict[genres_dict["recommended_type"] == genre]
    # Drop the 'recommended_type' column
    popular_movies = popular_movies.drop('recommended_type', axis=1)
    # Resetting the indices
    popular_movies = popular_movies.reset_index(drop=True)
    
    return popular_movies

# test = get_popular_movies("Adventure")