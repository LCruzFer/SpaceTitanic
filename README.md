# README

## Goals
I am using the kaggle challenge "Spaceship Titanic" to learn about LightGBM, Spark, Airflow and improve my skills in git.

## Task
The task of the Spaceship Titanic challenge is to predict which passengers of the Space Titanic are transported to another transmission based on the provided information about each passenger. To create a prediction I use a boosted trees approach, more precisely the LigthGBM framework - next to contributing a solution to the challenge I use it to learn about LigthGBM and improve my skills in git 

## Preprocessing & New Features
To get a first simple model running I will keep preprocessing and feature engineering to a minimum and then iteratively improve the process.

Iteration 1:
    - remove duplicates from data
    - drop all rows containing missing values
    - use imputation for numerical values only and only use simple 'median' imputation
    - scale numerical features

Iteration 2:
    -