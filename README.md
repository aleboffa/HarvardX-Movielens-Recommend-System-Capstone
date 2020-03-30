# HarvardX-Movielens-Recommend-System-Capstone
AlexBoffa-Movielens-Recommendation-System-Project-HarvardX

Hi, this is my Machine Learning Recommendation System Project for the Capstone. 

I have serious trouble with separate_row() function with this large dataset (10 millon rows). Always gave me " out of memory error", so:

First, I did the project until "Regularized Movie + User effect model", which gave me the RMSE = 0.8648170 that meets our goal: RMSE < 0.86490
It is here:
https://github.com/aleboffa/HarvardX-Simplest-Movielens-Recommendation-Project

Second: I didn´t stop here, so I did the full project including "Regularized Movie + User + YearofRating + Genre effect model". 
To do this, I converted the Data Frames "edx" and "validation" to Data Tables, then I applied the function separate_rows (),
and after 2 hours the system gave me the perfectly separated genres, ready to be used in the new model. I decided to use the Data Tables
for the whole project, since they worked faster in almost all the computations that I had to carry out and never indicated "memory error".
Also, the final resul of the best model, was RMSE = 0.8626426, much better than the simple model.

Regards,

Alejandro Boffa  

Ethical Hacking University Expert 

https://www.linkedin.com/in/alejandro-oscar-boffa/
