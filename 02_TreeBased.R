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

# LOAD DATA ----------------
DATA_PATH <- "../bigdata/data/final_dataset_preprocessed_sample.csv" #TODO sample
df <- fread(DATA_PATH)

# TRAIN TEST SPLIT ---------------- #TODO evtl cross validation
df$id <- 1:nrow(df)
train <- df %>% sample_frac(.95)
test  <- anti_join(df, train, by = 'id')

dim(train)
dim(test)

# DECISION TREE --------------
dt <- rpart(Label ~ .,
            data = train,
            method = "class"
)
rpart.plot(dt)
dt$variable.importance

# Evaluation
pred <- predict(dt, test, type = 'class')
confusionMatrix(as.factor(pred), as.factor(test$Label), positive = "1")

# RANDOM FOREST ----------
rf <- randomForest(as.factor(Label) ~ ., data = train)

# Evaluation
pred <- predict(rf, newdata=test)
confusionMatrix(as.factor(pred), as.factor(test$Label), positive = "1")

# XG-BOOST ----------
train_mat <- as.matrix(train)
test_mat <- as.matrix(test)
train_dmat <- xgb.DMatrix(data = train_mat[,2:95], label = train_mat[,1])
test_dmat <- xgb.DMatrix(data = test_mat[,2:95], label = test_mat[,1])

xgb <- xgboost(data = train_dmat, # the data
               nround = 2, # max number of boosting iterations
               objective = "binary:logistic")  # the objective function

pred <- predict(xgb, newdata=test_dmat)
pred <- round(pred,0)
confusionMatrix(as.factor(pred), as.factor(test$Label), positive = "1")








# GRIDSEARCH -------
mod <- function(...) {
  rpart(Label ~ ., data = train, control = rpart.control(...))
}

gs <- list(minsplit = c(10, 20, 30),
           maxdepth = c(5, 10, 30, 50)) %>%
  cross_df()

gs <- gs %>% mutate(fit = pmap(gs, mod))
