###############################
# IMPORTANT: Excecuting this project with R 3.5.3 took me almost 3 hours,
# with Microsoft R Open 3.5.3 took me 20 minutes !!!
# take a look:
# https://mran.microsoft.com/documents/rro/multithread#mt-bench
# https://mran.microsoft.com/rro#resources


################################
# Create edx set, validation set
################################

# Note: with R 3.5.3, this process took my computer about 10 min create edx and validation sets,
# but after months waiting hours to do a process, for example 2 hours separate_rows (),
# I could process the entire file in 10 minutesafter I installed Microsoft R Open 3.5.3

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
set.seed(1) # if you are using R 3.5 or Microsoft R Open 3.5.3
# set.seed(1, sample.kind="Rounding") if using R 3.5.3 or later

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

#####################################################################
#####   Starting MovieLens Recomendation System Project
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


#########################################################################
# Data Pre-processing

#########################################################################
# Separate genre from datasets by "|" and yearofRating to view statistics
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

# Split in half Data Table "edxDT" 

index1 <- as.integer(nrow(edxDT) / 2)
index2 <- nrow(edxDT) - index1
edx1 <- edxDT[1:index1,]
edx2 <- edxDT[index2:nrow(edx),]

# separate "genres" in both data tables generated "edx1" & "edx2".
# It took 2 hours with R 3.5.3 and 2 minutes with Microsoft R Open 3.5.3
# Yes, procesing  ~24millon rows with Microsoft R Open in 2 minutes!!!!

edx1  <- edx1  %>% separate_rows(genres, sep = "\\|")
edx2  <- edx2  %>% separate_rows(genres, sep = "\\|")

# join both "edx_" data tables 

edxDT <- bind_rows(edx1, edx2) # now we have a dataset of ~24 million rows

# Extract yearofRating from timestamp in the dataset edxDT.
# We will use it as Year effect

edxDT <- as.data.table(edxDT %>% 
  mutate(yearofRating = as.integer(format(as.POSIXct(timestamp, origin="1970-01-01"), "%Y"))))

# removing big obsolete datasets

rm(edx1, edx2) # release memory Ram!

# removing "timestamp" column from both datasets

edxDT <- select(edxDT, -timestamp)


##############################################################################
# This is the RMSE function that will give us the scores found with our models.
# Our goal is obtain a RMSE < 0.86490

RMSE <- function(true_ratings, predicted_ratings) {
   sqrt(mean((true_ratings - predicted_ratings)^2))
}


########################################################
#  Spliting edx dataset in train_set and test_set
# ######################################################
#
#  The validation data should NOT be used for training the algorithm
#  and should ONLY be used for evaluating the RMSE of final algorithm.
#  You should split the edx data into separate training and test sets 
#  to design and test your algorithm.

set.seed(1) # if you are using R 3.5 or Microsoft R open 3.5.3
# set.seed(1, sample.kind="Rounding") # if using R 3.5 or later

test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# To make sure we don't include users and movies in the test set that do not appear in the training set,
# we remove these entries using the semi_join function:

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>% 
  semi_join(train_set, by = "userId")


######################################
# We start computing our models here #
######################################

############################
### Starting Naive model ###

# Calculating "just the average" of all movies

mu <- mean(train_set$rating)

# Calculating the RMSE on the test set

naive_rmse <- RMSE(test_set$rating, mu)

# Creating a results dataframe that will contains all RMSE results.
# Here we insert our first RMSE.

rmse_results <- data.frame(method = "Just the average", RMSE = naive_rmse)

###################################
### Starting Movie Effect Model ###

# Calculating the average by movie

movie_avgs <- train_set %>%
   group_by(movieId) %>%
   summarize(b_i = mean(rating - mu))

# Computing the predicted ratings on test dataset

predicted_ratings <- mu + test_set %>% 
   left_join(movie_avgs, by='movieId') %>%
   .$b_i

# Computing Movie effect model

model_1_rmse <- RMSE(predicted_ratings, test_set$rating)

# Adding the results to the rmse_results table

rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "Movie Effect Model",
                                     RMSE = model_1_rmse ))

##########################################
### Starting Movie + User Effect Model ###

# Calculating the average by user

user_avgs <- train_set %>%
   left_join(movie_avgs, by='movieId') %>%
   group_by(userId) %>%
   summarize(b_u = mean(rating - mu - b_i))

# Computing the predicted ratings on test dataset

predicted_ratings <- test_set %>%
   left_join(movie_avgs, by='movieId') %>%
   left_join(user_avgs, by='userId') %>%
   mutate(pred = mu + b_i + b_u) %>%
   .$pred

model_2_rmse <- RMSE(predicted_ratings, test_set$rating)

# Adding the results to the results dataset

rmse_results <- bind_rows(rmse_results,
                          data.frame(method = "Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))


##############################################
# Starting Regularization of our models.
##############################################

########################################
# Regularizing Movie + User Effect Model
# Computing the predicted ratings on test dataset using different values of lambda
# b_i is the Movie effect and b_u is User effect.
# Lambda is a tuning parameter.
# We are using cross-validation to choose the best lambda that minimize our RMSE.

lambdas <- seq(0, 10, 0.25)

# function rmses calculate predictions with several lambdas

rmses <- sapply(lambdas, function(l) {
   
   # Calculating the average by movie
  
   b_i <- train_set %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu) / (n() + l))
   
   # Calculating the average by user
   
   b_u <- train_set %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu) / (n() + l))
   
   # Computing the predicted ratings on test dataset
   
   predicted_ratings <- test_set %>%
      left_join(b_i, by = 'movieId') %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      .$pred
   
   # Predicting the RMSE on the test set
   
   return(RMSE(predicted_ratings, test_set$rating))
})

# Getting the best lambda value that minimize the RMSE on reg movie + user effects model

lambda <- lambdas[which.min(rmses)]

# We know that our best RMSE is given by: RMSE = min(rmses),
# but as a purpose of clarification,
# we compute again our estimate with best lambda found:

# Computing regularized estimates of b_i using best lambda

movie_avgs_reg <- train_set %>%
   group_by(movieId) %>%
   summarize(b_i = sum(rating - mu) / (n() + lambda), n_i = n())

# Computing regularized estimates of b_u using best lambda

user_avgs_reg <- train_set %>%
   left_join(movie_avgs_reg, by ='movieId') %>%
   group_by(userId) %>%
   summarize(b_u = sum(rating - mu - b_i) / (n() + lambda), n_u = n())

# Predicting ratings

predicted_ratings <- test_set %>%
   left_join(movie_avgs_reg, by = 'movieId') %>%
   left_join(user_avgs_reg, by = 'userId') %>%
   mutate(pred = mu + b_i + b_u) %>%
   .$pred

# Predicting the RMSE on the test set

model_3_rmse <- RMSE(predicted_ratings, test_set$rating)

# Adding the results to the rmse_results dataset

rmse_results <- bind_rows(rmse_results,
                       data.frame(method = "Regularized Movie + User Effects Model",
                                  RMSE = model_3_rmse ))

# Show the table with the diferents Models and their RMSEs results

rmse_results

######################################################
# final model RMSE results with edx and validation set
######################################################

# Computing regularized estimates of b_i using best lambda

movie_avgs_reg <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda), n_i = n())

# Computing regularized estimates of b_u using best lambda

user_avgs_reg <- edx %>%
  left_join(movie_avgs_reg, by ='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda), n_u = n())

# Predicting ratings

predicted_ratings <- validation %>%
  left_join(movie_avgs_reg, by = 'movieId') %>%
  left_join(user_avgs_reg, by = 'userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Predicting the RMSE on the validation set

model_final_rmse <- RMSE(predicted_ratings, validation$rating)

#############
# Our RMSEs found with train an test sets from splitted edx:

rmse_results # %>% knitr::kable()

#############
# Our Final RMSE with edx and validation datasets is:

model_final_rmse # %>% knitr::kable()


##############
# comparing 10 first predictions with true ratings

versus <- cbind("Predicted Rating" = predicted_ratings[1:10], "True Rating" = validation$rating[1:10])



###########################################################
###########################################################
# Remmember, our goal is obtain a RMSE < 0.86490
# And voila, running this project in R on my computer,
# I have obtained these rmse_results,
#
#                                  method              RMSE
#
# 1:                       Just the average       1.0600537
# 2:                     Movie Effect Model       0.9429615
# 3:             Movie + User Effects Model       0.8646844
# 4: Regularized Movie + User Effects Model       0.8641362
# 5: FINAL Regul Movie + User Effects Model       0.8648177
###########################################################
###########################################################
