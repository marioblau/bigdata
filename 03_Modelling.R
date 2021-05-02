#-------------------------------------------------------------------------------------
# 01) Load Data
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
library(tidyverse)
library(pryr)
library(data.table)

# Packages Trees
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(xgboost)

# Evaluation
library(pROC)
library(grid)

# LOAD DATA ----------------
DATA_PATH <- "data/final_dataset_preprocessed_sample.csv" #TODO sample
df <- fread(DATA_PATH)

# TRAIN TEST SPLIT ----------------
df$id <- 1:nrow(df)
train <- df %>% sample_frac(.95)
test  <- anti_join(df, train, by = 'id')
save(train, test, file = "results/train_test.RData")

# DECISION TREE --------------
dt <- rpart(Label ~ ., data = train, method = "class")
save(dt, file = "results/decisiontree.RData")

# RANDOM FOREST ----------
rf <- randomForest(as.factor(Label) ~ ., data = train)
save(rf, file = "results/randomforest.RData")

# XG-BOOST ----------
train_mat <- as.matrix(train)
test_mat <- as.matrix(test)
train_dmat <- xgb.DMatrix(data = train_mat[,2:95], label = train_mat[,1])
test_dmat <- xgb.DMatrix(data = test_mat[,2:95], label = test_mat[,1])

xgb <- xgboost(data = train_dmat, # the data
               nround = 2, # max number of boosting iterations
               objective = "binary:logistic")  # the objective function
save(xgb, file = "results/xgboost.RData")

# .... TODO add models Mario


