#-------------------------------------------------------------------------------------
# 03) Modelling
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())

suppressPackageStartupMessages({
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

  # Package parallel computing
  library(doParallel) #https://www.geeksforgeeks.org/random-forest-with-parallel-computing-in-r-programming/

  # runtime analysis
  library(profvis)
})

# LOAD DATA ----------------
#DATA_PATH <- "C:/Users/mario/Desktop/MBF_St.Gallen/2. Semester/Big_data_analytics/Dataset_02_05/final_dataset_preprocessed_sample_scaled.csv"
DATA_PATH <- "data/final_dataset_preprocessed_sample_scaled.csv"

df <- fread(DATA_PATH)

# TRAIN TEST SPLIT ----------------
df$id <- 1:nrow(df)
train <- df %>% sample_frac(.95)
test  <- anti_join(df, train, by = 'id')
#save(train, test, file = "results/train_test.RData")
save(train, test, file = "results/scaled/train_test.RData")


# DECISION TREE --------------
print(paste("Training Decision Tree", Sys.time()))

#without parallelization
system.time({
  dt <- rpart(Label ~ ., data = train, method = "class")
})

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))

system.time({
  dt <- rpart(Label ~ ., data = train, method = "class")
})
stopCluster(cl)

#save(dt, file = "results/decisiontree.RData")
save(dt, file = "results/scaled/decisiontree.RData")

# RANDOM FOREST ----------
print(paste("Training Random Forest", Sys.time()))

#without parallelization
system.time({
  rf <- randomForest(as.factor(Label) ~ ., data = train)
})

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))

system.time({
  rf <- randomForest(as.factor(Label) ~ ., data = train)
})
stopCluster(cl)

# save
save(rf, file = "results/scaled/randomforest.RData")


# XG-BOOST ----------
print(paste("Training XG-Boost", Sys.time()))

train_mat <- as.matrix(train)
test_mat <- as.matrix(test)
train_dmat <- xgb.DMatrix(data = train_mat[,-c(1)], label = train_mat[,1])
test_dmat <- xgb.DMatrix(data = test_mat[,-c(1)], label = test_mat[,1])

#without parallelization
system.time({
  xgb <- xgboost(data = train_dmat, # the data
                 nround = 2, # max number of boosting iterations
                 objective = "binary:logistic")  # the objective function
})

#with parallelization
  cl<-makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
system.time({
  xgb <- xgboost(data = train_dmat, # the data
                 nround = 2, # max number of boosting iterations
                 objective = "binary:logistic")  # the objective function
})
stopCluster(cl)

#save
save(xgb, file = "results/scaled/xgboost.RData")



# LOGISTIC REGRESSION --------
print(paste("Training Logistic Regression", Sys.time()))

#without parallelization
system.time({
  glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")
})

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))

system.time({
  glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")
})

stopCluster(cl)

save(glm, file = "results/scaled/logregression.RData")



# KNN --------
print(paste("Training KNN", Sys.time()))

#without parallelization
system.time({
  knn <- knn(train[,-c(1)],test[,-c(1)],cl=train$Label,k=3,prob=TRUE)
})

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))

system.time({
  knn <- knn(train[,-c(1)],test[,-c(1)],cl=train$Label,k=3,prob=TRUE)
})

stopCluster(cl)

# save
save(knn, file = "results/scaled/knn.RData")


# Naive Bayes ------------
print(paste("Training Naive Bayes", Sys.time()))

#without parallelization
system.time({
  nb <- naiveBayes(x = train[,-c(1)], y = train[,c(1)])
})

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))

system.time({
  nb <- naiveBayes(x = train[,-c(1)], y = train[,c(1)])
})
stopCluster(cl)

# save
save(nb, file = "results/scaled/nb.RData")



# SVM --------
print(paste("Training SVM", Sys.time()))
#without parallelization
system.time({
  svm <- svm(formula = Label ~ ., data = train, #### only 10 rows: train[1:10,]
             type = 'C-classification',
             kernel = 'linear')
})

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
system.time({
  svm <- svm(formula = Label ~ ., data = train, #### only 10 rows: train[1:10,]
             type = 'C-classification',
             kernel = 'linear')
})
stopCluster(cl)

# save
save(svm, file = "results/scaled/svm.RData")

