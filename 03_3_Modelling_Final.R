#-------------------------------------------------------------------------------------
# 03) Modelling (train models on entire DF)
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

  # Packages ML models (faster implementation)
  library(ranger)
  library(FNN)
  library(LiblineaR)

  # Package parallel computing
  library(doParallel) # https://www.geeksforgeeks.org/random-forest-with-parallel-computing-in-r-programming/

  # Packages runtime analysis
  library(profvis)
  library(htmlwidgets)  # requires to install pandoc! --> https://pandoc.org/installing.html
  library(tictoc)
  library(pryr)
})

SMPL_FRAC <- 1

DATA_PATH <- paste0("data/final_dataset_preprocessed_sample100_scaled.csv")

print(paste0("Start FINAL Modelling  on ", SMPL_FRAC*100, "% of data ", Sys.time()))

#p <- profvis({ # sample rate = 10ms
# LOAD DATA ----------------
print(paste("Loading data from: ",DATA_PATH))
df <- fread(DATA_PATH) %>% #TODO remove df again? chck with memchange
  sample_frac(.,SMPL_FRAC) # sample only if SAMPLE variable is true

# TRAIN TEST SPLIT ----------------
print(paste("Create train-test split...", Sys.time()))
df$id <- 1:nrow(df)
train <- df %>% sample_frac(.95)
test  <- anti_join(df, train, by = 'id')
rm(df)
print(paste("Save train-test split...", Sys.time()))
#save(train, test, file = paste0("results/scaled/train_test_sample", SMPL_FRAC*100,".RData"))


# DECISION TREE --------------
print(paste("Training Decision Tree", Sys.time()))

cl <- makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
dt <- rpart(Label ~ ., data = train, method = "class")
stopCluster(cl)

print(paste("Save Decision Tree", Sys.time()))
save(dt, file = paste0("results/scaled/decisiontree_sample", SMPL_FRAC*100,".RData"))
rm(dt)

# RANDOM FOREST ----------
print(paste("Training Random Forest", Sys.time()))

rf <- ranger(as.factor(Label) ~ .,
                data = train,
                probability = TRUE)

print(paste("Save Random Forest", Sys.time()))
save(rf, file = paste0("results/scaled/randomforest_sample", SMPL_FRAC*100,".RData"))
rm(rf)

# XG-BOOST ----------
print(paste("Training XG-Boost", Sys.time()))

train_mat <- as.matrix(train)
test_mat <- as.matrix(test)
train_dmat <- xgb.DMatrix(data = train_mat[,-c(1)], label = train_mat[,1])
test_dmat <- xgb.DMatrix(data = test_mat[,-c(1)], label = test_mat[,1])

cl <- makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
xgb <- xgboost(data = train_dmat, # the data
               nround = 2, # max number of boosting iterations
               objective = "binary:logistic")  # the objective function
stopCluster(cl)

print(paste("Save XG-Boost", Sys.time()))
save(xgb, file = paste0("results/scaled/xgboost_sample", SMPL_FRAC*100,".RData"))
rm(xgb, train_mat, test_mat, train_dmat, test_dmat)

# LOGISTIC REGRESSION --------
print(paste("Training Logistic Regression", Sys.time()))

glm <- LiblineaR(train[,-1], train$Label, type = 1) # type:   0 – L2-regularized logistic regression (primal)

print(paste("Save Logistic Regression", Sys.time()))
save(glm, file = paste0("results/scaled/logregression_sample", SMPL_FRAC*100,".RData"))
rm(glm)

# KNN --------
print(paste("Training KNN", Sys.time()))
knn <- FNN::knn(train[,-c(1)],test[,-c(1)],cl=train$Label,k=3,prob=TRUE)

print(paste("Save KNN", Sys.time()))
save(knn, file = paste0("results/scaled/knn_sample", SMPL_FRAC*100,".RData"))
rm(knn)

# Naive Bayes ------------
print(paste("Training Naive Bayes", Sys.time()))

cl <- makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
nb <- naiveBayes(x = train[,-c(1)], y = train[,c(1)])
stopCluster(cl)

print(paste("Save Naive Bayes", Sys.time()))
save(nb, file = paste0("results/scaled/nb_sample", SMPL_FRAC*100,".RData"))
rm(nb)

# SVM --------
print(paste("Training SVM", Sys.time()))
svm <- LiblineaR(train[,-1], train$Label, type = 1) # type:  1 – L2-regularized L2-loss support vector classification (dual)

print(paste("Save SVM", Sys.time()))
save(svm, file = paste0("results/scaled/svm_sample", SMPL_FRAC*100,".RData"))
rm(svm)
#})
#htmlwidgets::saveWidget(p, paste0("results/ProfVis/03_3_Modelling_Final_sample",SMPL_FRAC*100,".html"))
print(paste("Saved ProfVis Analysis!", Sys.time()))

# FINISH ------
print(paste("Finished script!", Sys.time()))
