#-------------------------------------------------------------------------------------
# 03) Modelling
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
library(tidyverse)
library(pryr)
library(data.table)

# Packages ML models
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(xgboost)
library(class)
library(e1071)



# LOAD DATA ----------------
#DATA_PATH <- "data/final_dataset_preprocessed_sample.csv" #TODO sample
DATA_PATH <- "C:/Users/mario/Desktop/MBF_St.Gallen/2. Semester/Big_data_analytics/Dataset_02_05/final_dataset_preprocessed_sample_scaled.csv" 
df <- fread(DATA_PATH)

# TRAIN TEST SPLIT ----------------
df$id <- 1:nrow(df)
train <- df %>% sample_frac(.95)
test  <- anti_join(df, train, by = 'id')
#save(train, test, file = "results/train_test.RData")
save(train, test, file = "results/scaled/train_test.RData")

# DECISION TREE --------------
dt <- rpart(Label ~ ., data = train, method = "class")
#save(dt, file = "results/decisiontree.RData")
save(dt, file = "results/scaled/decisiontree.RData")

# RANDOM FOREST ----------
rf <- randomForest(as.factor(Label) ~ ., data = train)
#save(rf, file = "results/randomforest.RData")
save(rf, file = "results/scaled/randomforest.RData")

# XG-BOOST ----------
train_mat <- as.matrix(train)
test_mat <- as.matrix(test)
train_dmat <- xgb.DMatrix(data = train_mat[,-c(1)], label = train_mat[,1])
test_dmat <- xgb.DMatrix(data = test_mat[,-c(1)], label = test_mat[,1])

xgb <- xgboost(data = train_dmat, # the data
               nround = 2, # max number of boosting iterations
               objective = "binary:logistic")  # the objective function
#save(xgb, file = "results/xgboost.RData")
save(xgb, file = "results/scaled/xgboost.RData")

# LOGISTIC REGRESSION --------
glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")
save(glm, file = "results/scaled/logregression.RData")

# KNN --------
knn <- knn(train[,-c(1)],test[,-c(1)],cl=train$Label,k=3,prob=TRUE)
save(knn, file = "results/scaled/knn.RData")

# Naive Bayes ------------
nb <- naiveBayes(x = train[,-c(1)], y = train[,c(1)])
save(nb, file = "results/scaled/nb.RData")

# SVM --------
svm <- svm(formula = Label ~ ., data = train[1:10,], #### only 10 rows
                 type = 'C-classification',
                 kernel = 'linear')
save(svm, file = "results/scaled/svm.RData")




