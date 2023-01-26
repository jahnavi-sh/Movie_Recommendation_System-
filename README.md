# Movie_Recommendation_System-

This document is written to provide aid in understanding the project.

Contents of the document - 
Understanding the problem statement 
About the dataset
Machine learning 
Types of machine learning models with examples 
NumPy library 
Pandas library 
Scikit-learn library 
Difflib 
Fixing missing values in the dataset 
Data preprocessing
Label encoding
Cosine similarity 

What is the problem statement for the machine learning algorithm ?
A movie recommendation system gives a personalized recommendation to the user. This recommendation is based on the past preferences of the user. With the emergence of OTT platforms like Netflix, Amazon Prime Video etc, machine learning algorithms are utilized to give user a personalized experience. As the more relevant products the user finds on the platform, the higher is their engagement resulting in longer subscription and greater revenue for the company. There are 3 types of recommendations. 
1. Content based recommendation system 
2. Popularity based recommendation system 
3. Collaborative Recommendation system 

In this project, our job is to build a movie recommendation system. Here, we will take input from the user - a movie of their choice. Then, give an output of the list of movies from the dataset that are similar to the input

More about the dataset - 
The dataset contains the following columns/features - 
index 
budget - budget for the movie production 
genres 
homepage - website where the movie is available to watch 
identification number for the movie 
keywords - keywords for the description of movie 
original_language - original language of the movie
original_title - title of the movie 
overview - a basic description to the storyline of the movie 
popularity - popularity score of the movie 
production_companies - media company which produced the movie  
production_countries - countries where the movie is shot 
release_date - release date of the movie in theatres and/or ott platforms 
revenue - total revenue/profit earned 
runtime - total duration of the movie 
spoken_languages
status - released or not released 
tagline - tagline of the movie 
title - title of the movie 
vote_average - average ratings/score of the movie 
vote_count - number of votes given to the movie 
cast - list of members of cast of the movie 
crew - list of supporting crew members in the production of movie 
director - director of the movie

The dataset consists of 4803 rows (4804 data points) and 24 columns 24 features as mentioned above).

Machine learning - 
Machine learning enables the processing of sonar signals and target detection. Machine Learning is a subset of Artificial Intelligence. This involves the development of computer systems that are able to learn by using algorithms and statistical measures to study data and draw results from it. Machine learning is basically an integration of computer systems, statistical mathematics and data.

Machine Learning is further divided into three classes - Supervised learning, Unsupervised learning and Reinforcement Learning. 

Supervised learning is a machine learning method in which models are trained using labelled data. In supervised learning, models need to find the mapping function and find a relationship between the input and output. In this, the user has a somewhat idea of what the output should look like. It is of two types - regression (predicts results with continuous output. For example, given the picture of a person, we have to predict their age on the basis of the given picture) and classification (predict results in a discrete output. For example, given a patient with a tumor, we have to predict whether the tumor is malignant or benign.) 

Unsupervised learning is a method in which patterns are inferred from the unlabelled input data. It allows us to approach problems with little or no idea what the results should look like. We can derive structure from the data where we don’t necessarily know the effect of variables. We can derive the structure by clustering the data based on relationships among the variables in the data. With unsupervised learning there is no feedback on the prediction results. It is of two types - clustering (model groups input data into groups that are somehow similar or related by different variables. For example, clustering data of thousands of genes into groups) and non-clustering (models identifies individual inputs. It helps us find structure in a chaotic environment. For example, the cocktail party problem where we need to identify different speakers from a given audiotape.)

Reinforcement learning is a feedback-based machine learning technique. It is about taking suitable action to maximise reward in a particular situation. For example, a robotic dog learning the movement of his arms or teaching self-driving cars how to depict the best route for travelling. 

In this project we are simply using a data preprocessing method to find similarity. We will use cosine similarity to build this movie recommendation system. 

Cosine similarity is a measure of similarity. It is often used to measure document similarity in text analysis. It calculates the cosine angle between the two vector lists. For cosine function the value varies from 1 to -1. For two overlapping vectors, the value of cosine will be maximum and minimum for two precisely opposite vectors. 

Formula - 
Similarity = (A.B) / (||A||.||B||) 
Here, A and B are vectors 

Libraries used in the project - 
NumPy  
It is a python library used for working with arrays. It has functions for working in the domain of linear algebra, fourier transform, and matrices. It is the fundamental package for scientific computing with python. NumPy stands for numerical python. 

NumPy is preferred because it is faster than traditional python lists. It has supporting functions that make working with ndarray very easy. Arrays are frequently used where speed and resources are very important. NumPy arrays are faster because it is stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently. This is locality of reference in computer science. 

Pandas - 
Pandas is made for working with relational or labelled data both easily and intuitively. It provides various data structures and operations for manipulating numerical data and time series. 

It has a lot of advantages like - 
Fast and efficient for manipulating and analyzing data
Data from different file objects can be loaded 
Easy handling of missing data in data preprocessing 
Size mutability 
Easy dataset merging and joining 
Flexible reshaping and pivoting of datasets 
Gives time-series functionality 

Pandas is built on top of NumPy library. That means that a lot of structures of NumPy are used or replicated in Pandas. The data produced by pandas are often used as input for plotting functions of Matplotlib, statistical analysis in SciPy, and machine learning algorithms in Scikit-learn. 

Scikit-Learn - 
It provides efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction. It has numerous machine learning, pre-processing, cross validation, and visualization algorithms. 

Difflib - 
Difflib is a module in python the consists of simple functions and classes that are used to compare datasets. It gives output of the sequence comparisons in a format that can be read by human, using delta to show the differences more efficiently. It is most commonly used to compare two string values. 

Missing values - 
Missing values are common when working with real-world datasets. Missing data could result from a human factor, a problem in electrical sensors, missing files, improper management or other factors. Missing values can result in loss of significant information. Missing value can bias the results of model and reduce the accuracy of the model. There are various methods of handling missing data but unfortunately they still introduce some bias such as favoring one class over the other but these methods are useful. 

In Pandas, missing values are represented by NaN. It stands for Not a Number. 

Reasons for missing values - 
Past data may be corrupted due to improper maintenance
Observations are not recorded for certain fields due to faulty measuring equipments. There might by a failure in recording the values due to human error. 
The user has not provided the values intentionally. 

Why we need to handle missing values - 
Many machine learning algorithms fail if the dataset contains missing values. 
Missing values may result in a biased machine learning model which will lead to incorrect results if the missing values are not handled properly. 
Missing data can lead to lack of precision. 

Types of missing data - 
Understanding the different types of missing data will provide insights into how to approach the missing values in the dataset. 
Missing Completely at Random (MCAR) 
There is no relationship between the missing data and any other values observed or unobserved within the given dataset. Missing values are completely independent of other data. There is no pattern. The probability of data being missing is the same for all the observations. 
The data may be missing due to human error, some system or equipment failure, loss of sample, or some unsatisfactory technicalities while recording the values.
It should not be assumed as it’s a rare case. The advantage of data with such missing values is that the statistical analysis remains unbiased.   
Missing at Random (MAR)
The reason for missing values can be explained by variables on which complete information is provided. There is relationship between the missing data and other values/data. In this case, most of the time, data is not missing for all the observations. It is missing only within sub-samples of the data and there is pattern in missing values. 
In this, the statistical analysis might result in bias. 
Not MIssing at Random (NMAR)
Missing values depend on unobserved data. If there is some pattern in missing data and other observed data can not explain it. If the missing data does not fall under the MCAR or MAR then it can be categorized as MNAR. 
It can happen due to the reluctance of people in providing the required information. 
In this case too, statistical analysis might result in bias. 

How to handle missing values - 

isnull().sum() - shows the total number of missing values in each columns 

We need to analyze each column very carefully to understand the reason behind missing values. There are two ways of handling values - 
Deleting missing values - this is a simple method. If the missing value belongs to MAR and MCAR then it can be deleted. But if the missing value belongs to MNAR then it should not be deleted. 
The disadvantage of this method is that we might end up deleting useful data. 
You can drop an entire column or an entire row. 
Imputing missing values - there are various methods of imputing missing values
Replacing with arbitrary value 
Replacing with mean - most common method. But in case of outliers, mean will not be appropriate
Replacing with mode - mode is most frequently occuring value. It is used in case of categorical features. 
Replacing with median - median is middlemost value. It is better to use median in case of outliers. 
Replacing with previous value - it is also called a forward fill. Mostly used in time series data. 
Replacing with next value - also called backward fill. 
Interpolation 

For this project, we replace the null values with null strings. 

Data preprocessing - 
The dataset contains textual data that the machine learning algorithms cannot interpret.  We need to preprocess the data before feeding it into the logistic regression model. 
Following are the data preprocessing method used for treating the data - 

TfidfVectorizer - 
Term frequency inverse document frequency
This is used for converting text into a meaningful representation of numbers which is used to fit machine learning algorithm for prediction. 

Tf-idf is a measure of originality of a word by comparing the number of times a word appears in a document with the number of documents the word appears in. count vectorizer give number of frequency with respect to index of vocabulary where as tf-idf consider overall documents of weight of words. 

Tf-ifd value increases proportionally to the number of times a word appears in the document and decreases with the number of documents in the corpus that contains the word. It consists of two parts - 
Term Frequency(TF) : Term frequency specifies how frequently a term appears in the entire document.It is the probability of finding a word within the document.It calculates the number of times a word occurs in a review, with respect to the total number of words in the review.
Inverse Document Frequency(IDF) : The inverse document frequency is a measure of whether a term is rare or frequent across the documents in the entire corpus. It highlights those words which occur in very few documents across the corpus, or in simple language, the words that are rare have high IDF score. IDF is a log normalised value, that is obtained by dividing the total number of documents D in the corpus by the number of documents containing the term t , and taking the logarithm of the overall term. 

fit_transform() method 
This method is used on training data so that we can scale the training data and also learn the scaling parameters of that data. The model learns the mean and variance of features of training set. These learned parameters are used to scale our test data. 
The fit method calculates mean and variance of each feature present in the data.the transform method transforms all the features using the respective mean and variance. 

Then we test the cosine similarity by finding the similarity score. 

Take input from the user. Take an input of the movies of the user’s choice. 
Put this into a list with the .tolist() method. 

get_close_match() - 
It is a tool that accepts parameters and return the closest matches to the target string. It accepts two four parameters or arguments. 

The first argument is the word that has to be targeted. 
Second argument is the array of variables or terms that point to an array of strings. 
Third argument enables the user to define a limit to the number of outputs that are returned. 
Fourth argument determines the similarity between two words need to be in order to be returned as an output.  

The list is then compared, sorted and the recommendation system gives a list of similar movies from the dataset as per the input given by the user. 
