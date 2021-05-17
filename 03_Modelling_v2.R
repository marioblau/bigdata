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

# TODO select sample size
SMPL_FRAC <- 1 # 0.01, 0.1, 0.2, ... 0.9, 1.0

DATA_PATH <- paste0("data/final_dataset_preprocessed_sample100_scaled.csv")

print(paste0("Start Modelling on ", SMPL_FRAC*100, "% of data ", Sys.time()))
p <- profvis({
  # LOAD DATA ----------------
  print(paste("Loading data from: ",DATA_PATH))
  df <- fread(DATA_PATH) %>%
    sample_frac(.,SMPL_FRAC) # sample only if SAMPLE variable is true

  # TRAIN TEST SPLIT ----------------
  print(paste("Create train-test split...", Sys.time()))
  df$id <- 1:nrow(df)
  train <- df %>% sample_frac(.95)
  test  <- anti_join(df, train, by = 'id')
  save(train, test, file = paste0("results/scaled/train_test_sample", SMPL_FRAC*100,".RData"))


  # DECISION TREE --------------
  print(paste("Training Decision Tree", Sys.time()))

  #without parallelization
  dt <- rpart(Label ~ ., data = train, method = "class")

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
  dt <- rpart(Label ~ ., data = train, method = "class")
  stopCluster(cl)

  save(dt, file = paste0("results/scaled/decisiontree_sample", SMPL_FRAC*100,".RData"))

  # RANDOM FOREST ----------
  print(paste("Training Random Forest", Sys.time()))
  #without parallelization
  rf <- randomForest(as.factor(Label) ~ ., data = train)

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
  rf <- randomForest(as.factor(Label) ~ ., data = train)
  stopCluster(cl)

  save(rf, file = paste0("results/scaled/randomforest_sample", SMPL_FRAC*100,".RData"))

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

  save(xgb, file = paste0("results/scaled/xgboost_sample", SMPL_FRAC*100,".RData"))

  # LOGISTIC REGRESSION --------
  print(paste("Training Logistic Regression", Sys.time()))

  #without parallelization
  glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")

  #with parallelization
  cl <- makePSOCKcluster(4)
  registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
  glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")
  stopCluster(cl)

  save(glm, file = paste0("results/scaled/logregression_sample", SMPL_FRAC*100,".RData"))


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
  save(knn, file = paste0("results/scaled/knn_sample", SMPL_FRAC*100,".RData"))


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
  save(nb, file = paste0("results/scaled/nb_sample", SMPL_FRAC*100,".RData"))


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
  save(svm, file = paste0("results/scaled/svm_sample", SMPL_FRAC*100,".RData"))
})

htmlwidgets::saveWidget(p, paste0("ProfVis/03_Modelling_sample",SMPL_FRAC*100,".html"))
print(paste("Saved ProfVis Analysis!", Sys.time()))
