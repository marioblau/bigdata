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
  library(doParallel) # https://www.geeksforgeeks.org/random-forest-with-parallel-computing-in-r-programming/

  # Packages runtime analysis
  library(profvis)
  library(htmlwidgets)  # requires to install pandoc! --> https://pandoc.org/installing.html
})

#DATA_PATH <- "C:/Users/mario/Desktop/MBF_St.Gallen/2. Semester/Big_data_analytics/Dataset_02_05/final_dataset_preprocessed_sample_scaled.csv"
DATA_PATH <- "data/final_dataset_preprocessed_scaled.csv"

p <- profvis({
  # LOAD DATA ----------------
  print(paste("Loading data...", Sys.time()))
  df <- fread(DATA_PATH)

  # TRAIN TEST SPLIT ----------------
  print(paste("Create train-test split...", Sys.time()))
  df$id <- 1:nrow(df)
  train <- df %>% sample_frac(.95)
  test  <- anti_join(df, train, by = 'id')
  save(train, test, file = "results/scaled/train_test_FullDF.RData")


  # DECISION TREE --------------
  print(paste("Training Decision Tree", Sys.time()))

  #without parallelization
  dt <- rpart(Label ~ ., data = train, method = "class")

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
  dt <- rpart(Label ~ ., data = train, method = "class")
  stopCluster(cl)

  save(dt, file = "results/scaled/decisiontree_FullDF.RData")

  # RANDOM FOREST ----------
  print(paste("Training Random Forest", Sys.time()))
  #without parallelization
  rf <- randomForest(as.factor(Label) ~ ., data = train)

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
  rf <- randomForest(as.factor(Label) ~ ., data = train)
  stopCluster(cl)

  save(rf, file = "results/scaled/randomforest_FullDF.RData")

  # XG-BOOST ----------
  print(paste("Training XG-Boost", Sys.time()))

  train_mat <- as.matrix(train)
  test_mat <- as.matrix(test)
  train_dmat <- xgb.DMatrix(data = train_mat[,-c(1)], label = train_mat[,1])
  test_dmat <- xgb.DMatrix(data = test_mat[,-c(1)], label = test_mat[,1])

  #without parallelization
  xgb <- xgboost(data = train_dmat, # the data
                 nround = 2, # max number of boosting iterations
                 objective = "binary:logistic")  # the objective function

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
  xgb <- xgboost(data = train_dmat, # the data
                 nround = 2, # max number of boosting iterations
                 objective = "binary:logistic")  # the objective function
  stopCluster(cl)

  save(xgb, file = "results/scaled/xgboost_FullDF.RData")

  # LOGISTIC REGRESSION --------
  print(paste("Training Logistic Regression", Sys.time()))

  #without parallelization
  glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
  glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")
  stopCluster(cl)

  save(glm, file = "results/scaled/logregression_FullDF.RData")


  # KNN --------
  print(paste("Training KNN", Sys.time()))
  #without parallelization
  knn <- knn(train[,-c(1)],test[,-c(1)],cl=train$Label,k=3,prob=TRUE)

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
  knn <- knn(train[,-c(1)],test[,-c(1)],cl=train$Label,k=3,prob=TRUE)
  stopCluster(cl)

  # save
  save(knn, file = "results/scaled/knn_FullDF.RData")


  # Naive Bayes ------------
  print(paste("Training Naive Bayes", Sys.time()))
  #without parallelization
  nb <- naiveBayes(x = train[,-c(1)], y = train[,c(1)])

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))

  nb <- naiveBayes(x = train[,-c(1)], y = train[,c(1)])
  stopCluster(cl)

  # save
  save(nb, file = "results/scaled/nb_FullDF.RData")


  # SVM --------
  print(paste("Training SVM", Sys.time()))
  #without parallelization
  svm <- svm(formula = Label ~ ., data = train, #### only 10 rows: train[1:10,]
             type = 'C-classification',
             kernel = 'linear')

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
  svm <- svm(formula = Label ~ ., data = train, #### only 10 rows: train[1:10,]
             type = 'C-classification',
             kernel = 'linear')
  stopCluster(cl)

  # save
  save(svm, file = "results/scaled/svm_FullDF.RData")
})

htmlwidgets::saveWidget(p, "ProfVis/03_Modelling_ProfVis_FullDF.html")
print(paste("Saved ProfVis Analysis!", Sys.time()))
