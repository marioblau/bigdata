#-------------------------------------------------------------------------------------
# 03) Modelling (test alternatives for slow models)
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

  # Track time
  library(tictoc)
})

SMPL_FRAC <- 0.01 # 0.01, 0.1, 0.2, ... 0.9, 1.0

DATA_PATH <- paste0("../data/final_dataset_preprocessed_sample100_scaled.csv")

print(paste0("Start training selected algorithms on ", SMPL_FRAC*100, "% of data ", Sys.time()))

p <- profvis({ # sample rate = 10ms
  # LOAD DATA ----------------
  print(paste("Loading data from: ",DATA_PATH))
  df <- fread(DATA_PATH) %>%
    sample_frac(.,SMPL_FRAC) # sample only if SAMPLE variable is true

  # TRAIN TEST SPLIT ----------------
  print(paste("Create train-test split...", Sys.time()))
  df$id <- 1:nrow(df)
  train <- df %>% sample_frac(.95)
  test  <- anti_join(df, train, by = 'id')


  # RANDOM FOREST ----------
  print(paste("Training Random Forest", Sys.time()))

  # randomForest
  rf <- randomForest(as.factor(Label) ~ ., data = train)
  rf <- randomForest(as.factor(Label) ~ ., data = train, ntree = 500)

  # ranger
  rf_ranger <- ranger(as.factor(Label) ~ ., data = train)
  rf_ranger <- ranger(as.factor(Label) ~ ., data = train, num.trees = 500)

  # LOGISTIC REGRESSION ----------
  print(paste("Training Logistic Regression", Sys.time()))

  glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")

  glm <- LiblineaR(train[,-1], train$Label, type = 1) # type:   0 – L2-regularized logistic regression (primal)

  # KNN ----------
  print(paste("Training KNN", Sys.time()))

  # class
  knn <- class::knn(train[,-1], test[, -1], cl=train$Label, k=3, prob=TRUE)

  # FNN
  knn <- FNN::knn(train[,-1], test[, -1], cl=train$Label, k=3, prob=TRUE)

  # SVM ----------
  print(paste("Training SVM", Sys.time()))

  svm <- e1071::svm(formula = Label ~ ., data = train, #### only 10 rows: train[1:10,]
         type = 'C-classification',
         kernel = 'linear')

  svm <- LiblineaR(train[,-1], train$Label, type = 1) # type:  1 – L2-regularized L2-loss support vector classification (dual)

})
htmlwidgets::saveWidget(p, paste0("results/ProfVis/03_2_Modelling_SlowModels_sample",SMPL_FRAC*100,".html"))
print(paste("Saved ProfVis Analysis!", Sys.time()))

# FINISH ------
print(paste("Finished script!", Sys.time()))
end.time <- Sys.time()


