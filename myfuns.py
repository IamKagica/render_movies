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

# Flag to use smaller CSV file
S0_flag = True

# Preparing a smaller version of the R matrix
def make_small_R():
    
    # Load the CSV file containing R into a DataFrame
    R = pd.read_csv("Rmat.csv")
    
    # Create a new DataFrame with the top 3 rows
    Rsmall = R.head(3).copy()
    Rsmall.to_csv('Rsmall.csv', index=True)
    
# To make S0 if needed
Rsmall_path = "Rsmall.csv"
if os.path.exists(Rsmall_path):
    Rsmall = pd.read_csv(Rsmall_path)
else:
    make_small_R()
    # Saving and Loading Dataframe
    Rsmall_ = pd.read_csv(Rsmall_path)
#####

# Preparing a smaller CSV file involving only 
# 100 columns to account for limited RAM on hosting site.
# Getting indices of the 100 columns with the most ratings.
# Storing the dataframe in separate csv file.

def make_small_S0():
    
    # Load the CSV file containing R into a DataFrame
    R = pd.read_csv("Rsmall.csv")
    
    full_top30 = pd.read_csv('Symmetry_top30.csv')

    # Step 1: Count the number of NaN entries in each column
    nan_counts = full_top30.isna().sum(axis=0)
    
    # Step 2: Get the names of the top 100 columns with the fewest NaN entries
    top_columns = nan_counts.nsmallest(100).index
    
    # Step 3: Select the columns from the DataFrame
    limited_top30 = full_top30[top_columns]
    
    # Get the indices corresponding to movies dataframe
    
    most_rated_m = []
    
    for i in range(len(top_columns)):
    
        # If you have the index of a column, you can get its name
        column_index = int(top_columns[i])  # Getting the desired index
        column_name_at_index = R.columns[column_index]
        most_rated_m.append(column_name_at_index) 
        
    # Get the titles of the top ten scoring movies
    # Making sure that we keep the same structure as before
    most_rated_movies = []
    for i in range(len(most_rated_m)):
        for j in range(len(movies)):
            if (most_rated_m[i] == ('m' + str(movies.iloc[j].loc['movie_id']))):
                most_rated_movies.append(j)
                
    np.save("most_rated_movies", most_rated_movies)
    np.save("column_index", column_index)
    
    # Save the similarity matrix to a CSV file
    limited_top30.to_csv('S0_top30.csv', index=False)
        
# To make S0 if needed
S0_path = "S0_top30.csv"
most_rated_movies_path = "most_rated_movies.npy"
if os.path.exists(S0_path) and os.path.exists(most_rated_movies_path):
    S0 = pd.read_csv(S0_path)
else:
    make_small_S0()
    # Saving and Loading Dataframe
    S0 = pd.read_csv(S0_path)
#####

def movieID_match_columnIndex():
    
    # Create an empty DataFrame with specified columns
    columns = ['movie_id', 'col_id']
    movie2col = pd.DataFrame(columns=columns)

    # Load the CSV file containing R into a DataFrame
    R = pd.read_csv("Rsmall.csv")
    
    for i in range(len(R.columns)):
        for j in range(len(movies)):
            if (R.columns[i] == ('m' + str(movies.iloc[j].loc['movie_id']))):
                
                thisMovieID = int(movies.iloc[j].loc['movie_id'])
                thiscolID = i
                
                # New row to be added
                new_row = {'movie_id': thisMovieID, 'col_id': thiscolID}
                
                # Add the new row using loc
                movie2col.loc[len(movie2col)] = new_row
                
                break;
                    
    # Save the similarity matrix to a CSV file
    movie2col.to_csv('movie2col.csv', index=True)  
    
# To match column and ID
movie2col_path = "movie2col.csv"
if os.path.exists(movie2col_path):
    movie2col = pd.read_csv(movie2col_path)
else:
    movieID_match_columnIndex()
    # Saving and Loading Dataframe
    movie2col = pd.read_csv(movie2col_path)
#####

def top100_match_columnIndex():
    
    # Create an empty DataFrame with specified columns
    columns = ['col_id', 'hundred_id']
    col2hundred = pd.DataFrame(columns=columns)
        
    # Load the CSV file containing 100 columns into a DataFrame
    S0_100_col = pd.read_csv("S0_top30.csv")
    
    num_columns = np.shape(S0_100_col)[1]
    
    for i in range(num_columns):
                
        col_id = int(S0_100_col.columns[i])
        hundred_id = i
        
        # New row to be added
        new_row = {'col_id': col_id, 'hundred_id': hundred_id}
        
        # Add the new row using loc
        col2hundred.loc[len(col2hundred)] = new_row
                            
    # Save the similarity matrix to a CSV file
    col2hundred.to_csv('col2hundred.csv', index=False) 
    
# To match column and ID
col2hundred_path = "col2hundred.csv"
if os.path.exists(col2hundred_path):
    col2hundred = pd.read_csv(col2hundred_path)
else:
    top100_match_columnIndex()
    # Saving and Loading Dataframe
    col2hundred = pd.read_csv(col2hundred_path)
##### 

def get_displayed_movies():
    
    if (S0_flag == True):
        movies_index = np.load("most_rated_movies.npy")
        return movies.iloc[movies_index, :]
    else:
        return movies.head(100)

def dict_to_df(new_user_dict):
    
    # Load the CSV file containing R into a DataFrame
    R = pd.read_csv("Rsmall.csv")
    
    # Load the CSV containing the conversion from movie index
    # to the index of the Similarity matrix
    movie2col = pd.read_csv("movie2col.csv")
    
    # Copy a DataFrame
    unew_ratings = R.iloc[0, :]
    # Replace all values with NaN using numpy.nan
    unew_ratings = unew_ratings * np.nan
        
    # Loop over all keys in the dictionary
    for key in new_user_dict:
        i_index = movie2col[movie2col['movie_id'] == key]['col_id']
        unew_ratings.iloc[i_index] = new_user_dict[key]
        
    return unew_ratings

def small_df(df_full):
    
    # Load the CSV containing the positions 
    # for only 100 movies. 
    col2hundred = pd.read_csv("col2hundred.csv")
    
    numCols = 100
    
    # Create a new DataFrame with the first 100 columns
    small_ratings = df_full.iloc[:numCols].copy()
    # Replace all values with NaN using numpy.nan
    small_ratings = small_ratings * np.nan
            
    # Loop over all keys in the dictionary
    for i in range(numCols):
        thisRatingIndex = col2hundred[col2hundred['hundred_id'] == i]['col_id']
        thisRating = df_full[thisRatingIndex]
        small_ratings.iloc[i] = thisRating
        
    return small_ratings

# Testing
"""
def get_recommended_movies(new_user_ratings):
    
    # Convert the dictionary to a DataFrame if needed
    # Check if the variable is a dictionary
    if isinstance(new_user_ratings, dict):
        newuser = dict_to_df(new_user_ratings)
    else:
        newuser = new_user_ratings
        
    if(S0_flag == True):
        newuser = small_df(newuser)
        
    # Using IBCF function
    
    return movies.head(10)
"""
    
def get_recommended_movies(new_user_ratings):
    
    # Convert the dictionary to a DataFrame if needed
    # Check if the variable is a dictionary
    if isinstance(new_user_ratings, dict):
        newuser = dict_to_df(new_user_ratings)
    else:
        newuser = new_user_ratings
    
    # Testing #
    """
    newuser.iloc[2429] = 1
    newuser.iloc[2466] = 4
    newuser.iloc[1924] = 5
    newuser.iloc[1453] = 5
    newuser.iloc[2409] = 5
    """
        
    if(S0_flag == True):
        newuser = small_df(newuser)
                
    # print(np.nansum(newuser))

    # Using IBCF function
    
    # Loading similarity matrix
    if(S0_flag == True):
        S = pd.read_csv('S0_top30.csv')
    else:
        S = pd.read_csv('Symmetry_top30.csv')
    
    if(S0_flag == True):
        S_numrows = pd.read_csv('S0_top30.csv', usecols=[0])
    else:
        S_numrows = pd.read_csv('Symmetry_top30.csv', usecols=[0])
    
    # Load the CSV file containing R into a DataFrame
    R = pd.read_csv("Rsmall.csv")
    
    num_movies = np.shape(S_numrows)[0]
    # num_movies = np.shape(S)[0]
    # num_movies = (S)[0]
    list_scores = np.zeros(num_movies)
    
    # print(newuser)
    
    # Calculating predictions
    for l in range(num_movies):
        
        """
        jump = 500
        # Reading five hundred lines at a time
        if (l % jump == 0):
            subtract = l
            chunks = pd.read_csv("Symmetry_top30.csv", skiprows=l, nrows=jump, chunksize=jump//10)
            S_partial = pd.concat(chunks)
        """
                    
        Sl = S.iloc[l, :]
        
        # Sl = S_partial.iloc[l - subtract, :]
        Sl_w = Sl.to_numpy() * newuser.to_numpy()
        numerator = np.nansum(Sl_w)
        
        Sl_1 = Sl.to_numpy() * (newuser.to_numpy() / newuser.to_numpy())
        denominator = np.nansum(Sl_1)
        
        # Check if the object is a dictionary
        if isinstance(new_user_ratings, dict):
            if new_user_ratings.get(l) is not None:
                # Will set movies that have already been seen as lowest priority.
                list_scores[l] = -1
            elif (np.isnan(denominator) == True or denominator == 0):
                # If there aren't enough predictions, will set to zero. 
                list_scores[l] = 0
            else:
                score_pred = (numerator / denominator)
                list_scores[l] = score_pred
        else:
            if (np.isnan(newuser.iloc[l]) == False):
                # Will set movies that have already been seen as lowest priority.
                list_scores[l] = -1
            elif (np.isnan(denominator) == True or denominator == 0):
                # If there aren't enough predictions, will set to zero. 
                list_scores[l] = 0
            else:
                score_pred = (numerator / denominator)
                list_scores[l] = score_pred
        # Testing    
        """        
        if (l == 5):
            print("Predicted Score is")
            print(score_pred)
        """
                                            
    # Get indices of the top 10 largest values
    top_indices = np.argsort(list_scores)[-10:][::-1]
    
    # Testing
    # print(top_indices)
    # print(list_scores[top_indices])
    # print(R.columns[5])
    
    unew_movie_nums = []
        
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

# Testing
# dict_test = {2494: 2}
# dict_test = {2494: 2, 962: 1, 1160: 1, 2904: 1, 214: 1, 2221: 1, 2981: 1, 2830: 1, 960: 1, 2175: 1, 3853: 1, 2503: 1, 3374: 1, 874: 1, 1026: 1, 3306: 1, 626: 1, 755: 1, 1531: 1, 649: 1, 2963: 1, 3292: 1, 2905: 1, 167: 1, 2326: 1, 3880: 1, 1412: 1, 2197: 1, 1872: 1, 40: 1, 3025: 1, 669: 1, 2211: 1, 228: 1, 3495: 1, 331: 1, 298: 1, 3796: 1, 3803: 1, 1406: 1, 570: 1, 1455: 1, 3311: 1, 3410: 1, 3092: 1, 3532: 1, 726: 1, 1076: 1, 1144: 1, 96: 1, 131: 1, 1675: 1, 189: 1, 2493: 1, 3574: 1, 957: 1, 3239: 1, 1501: 1, 2545: 1, 2765: 1, 3492: 1, 49: 1, 961: 1, 525: 1, 554: 1, 129: 1, 200: 1, 3402: 1, 394: 1, 2101: 1, 2737: 1, 2865: 1, 359: 1, 3026: 1, 974: 1, 3912: 1, 1490: 1, 2215: 1, 2979: 1, 3579: 1, 563: 1, 992: 1, 1567: 1, 2189: 1, 2426: 1, 2839: 1, 3924: 1, 769: 1, 2537: 1, 3140: 1, 128: 1, 703: 1, 1138: 1, 1666: 1, 2536: 1, 3319: 1, 3434: 1, 3670: 1, 3884: 1}
# test = get_recommended_movies(dict_test)
# print(test["title"])

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
    R = pd.read_csv("Rsmall.csv")
    
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