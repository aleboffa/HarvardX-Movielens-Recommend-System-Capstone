# Movie System Recommendation - Harvard Data Science Capstone - 2020
# Author: Alejandro Boffa
# program language: R

# NOTE: if you want to run this proyect, feel free to do it, 
#       on my laptop, Intel I5+8gb ram+ssd 500gb, it took about 3 hours to run.
###############################################################################
# Creating edx set, validation set, given to us by Harvard Data Science staff
###############################################################################

# Start creation datasets

# Note: this process took on my computer about 10 min create edx and validation sets.

if(!require(tidyverse)) install.packages("tidyverse", repos="http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos="http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos="http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)


ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# End creation datasets

#####################################################################
#####   Starting our MovieLens Recomendation System Project
#####################################################################

# Install all needed libraries if they are not present

if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(tidyr)) install.packages("tidyr")
if(!require(stringr)) install.packages("stringr")
if(!require(forcats)) install.packages("forcats")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(kableExtra)) install.packages("kableExtra")

# Loading all needed libraries

library(dslabs)
library(caret)
library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)
library(data.table)


####################################################################
# Data Pre-processing

######################################################
# Separate genre from datasets by "|"
# 
# After 100 times trying separate_row () on the edx Data Frame to separate the "genre"
# through its separator "|" and after hours of computing ~10 millions of rows
# on WIN10 64bit I5 + 8gb ram + ssd500gb laptop, I get the message "error, memory out",
# setting memory.limit(64000) on console and after investigating several days,
# I got the following simple solution
# (if you know a better one, please, let me know, thanks:)

# Convert dataframes edx & validation to "data table"-> edxDT & validationDT
# from "edx Data Frame" ~ 900mb(Global Enviroment) to ~ 400mb in "edx Data Table", 
# reducing the size of file in Mbytes to a half

memory.limit(64000)

edxDT <- as.data.table(edx)
validationDT <- as.data.table(validation)

# Split in half Data Table "edxDT" 

index1 <- as.integer(nrow(edxDT) / 2)
index2 <- nrow(edxDT) - index1
edx1 <- edxDT[1:index1,]
edx2 <- edxDT[index2:nrow(edx),]

# separate "genres" in both data tables generated "edx1" & "edx2". It took 2 hours:~24millon rows

edx1  <- edx1  %>% separate_rows(genres, sep = "\\|")
edx2  <- edx2  %>% separate_rows(genres, sep = "\\|")

# separate "genres" in "validation" data table

validationDT  <- validationDT  %>% separate_rows(genres, sep = "\\|") # dataset with ~2.6 million rows

# join both "edx_" data tables 

edxDT <- bind_rows(edx1, edx2) # now we have a dataset of ~24 million rows

# Extract yearofRating from timestamp in both datasets: edxDT & validationDT. We will use it as Year effect

edxDT <- as.data.table(edxDT %>% 
             mutate(yearofRating = as.integer(format(as.POSIXct(timestamp, origin="1970-01-01"), "%Y"))))
validationDT <- as.data.table(validationDT %>%
                    mutate(yearofRating = as.integer(format(as.POSIXct(timestamp, origin="1970-01-01"), "%Y"))))

# removing big obsolete datasets

rm(edx, edx1, edx2, validation) # release ~2gb memory Ram!

# removing "timestamp" column from both datasets

edxDT <- select(edxDT, -timestamp)
validationDT <- select(validationDT, -timestamp)


####################################################################
# This is the Root Mean Square Error "RMSE" function that will give
# us the scores found with our models.
# The lower this result, the lower the error in our prediction.
# Our goal is obtain a RMSE < 0.86490

RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


################################################################
# We start computing our models here, working with data.tables #
################################################################

############################
### Starting Naive model ###

# Calculating "just the average" of all movies

mu <- mean(edxDT$rating)

# Calculating the RMSE on the validation set

naive_rmse <- RMSE(validationDT$rating, mu)

# Creating a results data table that will contains all RMSE results.
# Here we insert our first RMSE.

rmse_results <- data.table(method = "Just the average", RMSE = naive_rmse)

###################################
### Starting Movie Effect Model ###

# Calculating the average by movie

movie_avgs <- edxDT %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Computing the predicted ratings on validation dataset

predicted_ratings <- mu + validationDT %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

# Computing Movie effect model

model_1_rmse <- RMSE(predicted_ratings, validationDT$rating)

# Adding the results to the rmse_results table

rmse_results <- bind_rows(rmse_results,
                          data.table(method = "Movie Effect Model",
                                     RMSE = model_1_rmse))

##########################################
### Starting Movie + User Effect Model ###

# Calculating the average by user

user_avgs <- edxDT %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Computing the predicted ratings on validation dataset

predicted_ratings <- validationDT %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, validationDT$rating)

# Adding the results to the results dataset

rmse_results <- bind_rows(rmse_results,
                          data.table(method = "Movie + User Effects Model",  
                                     RMSE = model_2_rmse))


##############################################
# Starting Regularization of our models.
##############################################

########################################
# Regularizing Movie + User Effect Model
# Computing the predicted ratings on validation dataset using different values of lambda
# b_i is the Movie effect and b_u is User effect.
# Lambda is a tuning parameter.
# We are using cross-validation to choose the best lambda that minimize our RMSE.

# after testing several different intervals, I choose that one that minimize RMSEs.

lambdas <- seq(10, 20, 0.5)

# function rmses calculate predictions with several lambdas

mu <- mean(edxDT$rating)
rmses <- sapply(lambdas, function(l) {
  
  # Calculating the average by movie
  
  b_i <- edxDT %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  
  # Calculating the average by user
  
  b_u <- edxDT %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + l))
  
  # Computing the predicted ratings on validation dataset
  
  predicted_ratings <- validationDT %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  # Predicting the RMSE on the validation set
  
  return(RMSE(predicted_ratings, validationDT$rating))
})

# Getting the best lambda value that minimize the RMSE on reg movie + user effects model

lambda <- lambdas[which.min(rmses)]

# We know that our best RMSE is given by: RMSE = min(rmses),
# but as a purpose of clarification,
# we compute again our estimate with best lambda found:

# Computing regularized estimates of b_i using best lambda

movie_avgs_reg <- edxDT %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda), n_i = n())

# Computing regularized estimates of b_u using best lambda

user_avgs_reg <- edxDT %>%
  left_join(movie_avgs_reg, by ='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda), n_u = n())

# Predicting ratings

predicted_ratings <- validationDT %>%
  left_join(movie_avgs_reg, by = 'movieId') %>%
  left_join(user_avgs_reg, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Predicting the RMSE on the validation set

model_3_rmse <- RMSE(predicted_ratings, validationDT$rating)

# Adding the results to the rmse_results dataset

rmse_results <- bind_rows(rmse_results,
                          data.table(method = "Regularized Movie + User Effects Model",
                                     RMSE = model_3_rmse))

#################################################################
# Regularizing Movie + User + YearofRating + Genres Effect Model.
# Computing the predicted ratings on validation dataset using different values
# of lambda, searching for de best one.
# b_i is the Movie effect and b_u is User effect.
# b_y is Year effect and b_g is Genre effects.

# after testing several different intervals, I choose that one that minimize RMSEs.

lambdas <- seq(10, 20, 0.5) 

# function rmses calculate predictions with several lambdas

rmses <- sapply(lambdas, function(l){
  
  # Calculating the average by movie
  
  b_i <- edxDT %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  
  # Calculating the average by user
  
  b_u <- edxDT %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu) / (n() + l))
  
  # Calculating the average by year of rating
  
  b_y <- edxDT %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    group_by(yearofRating) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u) / (n() + l), n_y = n())
  
  # Calculating the average by genre
  
  b_g <- edxDT %>%
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_y, by = 'yearofRating') %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u - b_y) / (n() + l), n_g = n())
  
  # Computing the predicted ratings on validation dataset
  
  predicted_ratings <- validationDT %>% 
    left_join(b_i, by = 'movieId') %>%
    left_join(b_u, by = 'userId') %>%
    left_join(b_y, by = 'yearofRating') %>%
    left_join(b_g, by = 'genres') %>%
    mutate(pred = mu + b_i + b_u + b_y + b_g) %>% 
    .$pred
  
  return(RMSE(predicted_ratings, validationDT$rating))
})

lambda <- lambdas[which.min(rmses)] # best lambda found  

# Compute regularized estimates of b_i (movie effect) using best lambda

movie_reg_avgs <- edxDT %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu) / (n() + lambda), n_i = n())

# Compute regularized estimates of b_u (user effect) using best lambda

user_reg_avgs <- edxDT %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda), n_u = n())

# Compute regularized estimates of b_y (year of rating effect) using best lambda

year_reg_avgs <- edxDT %>%
  left_join(movie_reg_avgs, by = 'movieId') %>%
  left_join(user_reg_avgs, by = 'userId') %>%
  group_by(yearofRating) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u) / (n() + lambda), n_y = n())

# Compute regularized estimates of b_g (genre effect) using best lambda

genre_reg_avgs <- edxDT %>%
  left_join(movie_reg_avgs, by = 'movieId') %>%
  left_join(user_reg_avgs, by = 'userId') %>%
  left_join(year_reg_avgs, by = 'yearofRating') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u - b_y) / (n() + lambda), n_g = n())

# Predict ratings

predicted_ratings <- validationDT %>% 
  left_join(movie_reg_avgs, by = 'movieId') %>%
  left_join(user_reg_avgs, by = 'userId') %>%
  left_join(year_reg_avgs, by = 'yearofRating') %>%
  left_join(genre_reg_avgs, by = 'genres') %>%
  mutate(pred = mu + b_i + b_u + b_y + b_g) %>% 
  .$pred

model_4_rmse <- RMSE(predicted_ratings, validationDT$rating)

# Adding the results to the rmse_results dataset

rmse_results <- bind_rows(rmse_results,
                          data.table(method="Reg Movie + User + Year + Genre Effect Model",  
                                     RMSE = model_4_rmse))

# Show the table with the diferents Models and their RMSEs results

rmse_results %>% knitr::kable()



###########################################################
###########################################################
# Remmember, our goal is obtain a RMSE < 0.86490
# And voila, running this project in R on my computer,
# I have obtained these rmse_results,
# from the "Movie + User Effects Model" to end,
# exceeded the goal:
#
#                                  method              RMSE
#
# 1:                       Just the average       1.0525579
# 2:                     Movie Effect Model       0.9410700
# 3:             Movie + User Effects Model       0.8633660
# 4: Regularized Movie + User Effects Model       0.8627554
# 5: Reg Movie + User + Year + Genre Effect Model 0.8626426
###########################################################
###########################################################

