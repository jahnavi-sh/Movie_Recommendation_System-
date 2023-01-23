#First, understanding the problem statement - 
#A movie recommendation system gives a personalized recommendation to the user. This recommendation is based on the past preferences 
#of the user. With the emergence of OTT platforms like Netflix, Amazon Prime Video etc, machine learning algorithms are utilized 
#to give user a personalized experience. As the more relevant products the user finds on the platform, the higher is their 
#engagement resulting in longer subscription and greater revenue for the company. There are 3 types of recommendations. 
#1. Content based recommendation system 
#2. Popularity based recommendation system 
#3. Collaborative Recommendation system 

#In this project, our job is to build a movie recommendation system. Here, we will take input from the user - a movie of their choice. 
#Then, give an output of the list of movies from the dataset that are similar to the input.  

#Workflow for the system - 
#1. Data collection
#2. Data preprocessing 
#3. Feature extraction 
#4. Take user input 
#5. Cosine similarity 
#6. Give list of movies recommended 

#Import libraries 
#linear algebra - to build matrices 
import numpy as np 

#data preprocessing and exploration 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer

import difflib 
from sklearn.metrics.pairwise import cosine_distances

#data collection 
#loading the data from csv file to pandas dataframe 
movies_data = pd.read_csv(r'movies.csv')

#view the first five rows of the dataframe 
movies_data.head()

#The dataset contains the following columns/features - 
#1. index 
#2. budget - budget for the movie production 
#3. genres 
#4. homepage - website where the movie is available to watch 
#5. id - identification number for the movie 
#6. keywords - keywords for the description of movie 
#7. original_language - original language of the movie
#8. original_title - title of the movie 
#9. overview - a basic description to the storyline of the movie 
#10.popularity - popularity score of the movie 
#11.production_companies - media company which produced the movie  
#12.production_countries - countries where the movie is shot 
#13.release_date - release date of the movie in theatres and/or ott platforms 
#14.revenue - total revenue/profit earned 
#15.runtime - total duration of the movie 
#16.spoken_languages
#17.status - released or not released 
#18.tagline - tagline of the movie 
#19.title - title of the movie 
#20.vote_average - average ratings/score of the movie 
#21.vote_count - number of votes given to the movie 
#22.cast - list of members of cast of the movie 
#23.crew - list of supporting crew members in the production of movie 
#24.director - director of the movie

#view the total number of rows and columns - 
movies_data.shape 
#there are 4803 rows (4804 data points) and 24 columns (24 features)

#selecting the relevant features for recommendation 
selected_features = ['genres','keywords','tagline','cast','director']
#if you want you can take more features 

#replacing the null values with null strings
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

#combining all the 5 selected features 
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

#converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

#cosine similarity 
#getting the similarity scores using cosine similarity 
similiarity = cosine_similarity(feature_vectors)

#getting the moviw name from the user 
movie_name = input('Enter your movie of choice - ')

#getting a list with all the movie names given in the dataset 
list_of_all_title = movies_data['title'].tolist()

#finding the close match for the movie name given by the user 
find_close_match = difflib.get_close_match(movie_name, list_of_all_title)
print (find_close_match)
#this prints the movie recommended by the code based on the input given by the user 

#get the index of the movie 
index_of_the_movie = movies_data[movies_data.title == close_match]['Index'].values[0]
print (index_of_the_movie)

#getting a list of similar movies 
similiarity_score = list(enumerate(similarity[index_of_the_movie]))

#sorting the movies based on similarity score 
sorted_similar_movies = sorted(similiarity_score, key= lambda x:x[1], reverse = True)

#print the name of similar movies based on index of the movie 
print ('Movies recommended for you - \n')

i=1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index==index]['title'].values[0]
    if (i<30):
        print(i, '.', title_from_index)
        i+=1

