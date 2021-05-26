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
  library(biglasso)
  library(LiblineaR)
  library(parallelSVM)
  library(FNN)

  # Package parallel computing
  library(doParallel) # https://www.geeksforgeeks.org/random-forest-with-parallel-computing-in-r-programming/

  # Packages runtime analysis
  library(profvis)
  library(htmlwidgets)  # requires to install pandoc! --> https://pandoc.org/installing.html
  library(pryr)
})

SMPL_FRAC <- 0.01
DATA_PATH <- paste0("data/final_dataset_preprocessed_sample100_scaled.csv")
#load("data/train_test_sample100_seed2021.RData")

print(paste0("Start FINAL Modelling  on ", SMPL_FRAC*100, "% of data ", Sys.time()))

p <- profvis({ # sample rate = 10ms
# LOAD DATA ----------------
print(paste("Loading data from: ", DATA_PATH))
set.seed(2021)
df <- fread(DATA_PATH) %>%
  sample_frac(.,SMPL_FRAC, )

# TRAIN TEST SPLIT ----------------
print(paste("Create train-test split...", Sys.time()))
df$id <- 1:nrow(df)
set.seed(2021)
train <- df %>% sample_frac(.95)
test  <- anti_join(df, train, by = 'id')
rm(df)
print(paste("Save train-test split...", Sys.time()))
save(train, test, file = paste0("data/train_test_sample", SMPL_FRAC*100,".RData"))

# LOGISTIC REGRESSION --------
print(paste("Training Logistic Regression", Sys.time()))
xtrain_bigmat <- as.big.matrix(train[,-1])
ytrain_bigmat <- as.big.matrix(train$Label)

glm <- biglasso(xtrain_bigmat, ytrain_bigmat, family = "binomial", ncores = detectCores(all.tests = FALSE, logical = TRUE))

print(paste("Save Logistic Regression", Sys.time()))
save(glm, file = paste0("results/models/logregression_sample", SMPL_FRAC*100,".RData"))
rm(glm, xtrain_bigmat, ytrain_bigmat)

# Naive Bayes ------------
print(paste("Training Naive Bayes", Sys.time()))

ncores <- detectCores() #12
cl <- makeCluster(ncores) # Create cluster with desired number of cores:
registerDoParallel(cl) # Register cluster:
getDoParWorkers() # Find out how many cores are being used
nb <- train(Label ~ .,
            data = train,
            method = "nb")
stopCluster(cl)
registerDoSEQ()

print(paste("Save Naive Bayes", Sys.time()))
save(nb, file = paste0("results/models/nb_sample", SMPL_FRAC*100,".RData"))
rm(nb)

# DECISION TREE --------------
print(paste("Training Decision Tree", Sys.time()))

ncores <- detectCores() #12
cl <- makeCluster(ncores) # Create cluster with desired number of cores:
registerDoParallel(cl) # Register cluster:
getDoParWorkers() # Find out how many cores are being used
nb <- train(Label ~ .,
            data = train,
            method = "dt")
stopCluster(cl)
registerDoSEQ()

print(paste("Save Decision Tree", Sys.time()))
save(dt, file = paste0("results/models/decisiontree_sample", SMPL_FRAC*100,".RData"))
rm(dt)

# RANDOM FOREST ----------
print(paste("Training Random Forest", Sys.time()))

rf <- ranger(as.factor(Label) ~ .,
                data = train,
                importance = "impurity",
                probability = TRUE)

print(paste("Save Random Forest", Sys.time()))
save(rf, file = paste0("results/models/randomforest_sample", SMPL_FRAC*100,".RData"))
rm(rf)

# XG-BOOST ----------
print(paste("Training XG-Boost", Sys.time()))

ncores <- detectCores() #12
cl <- makeCluster(ncores) # Create cluster with desired number of cores:
registerDoParallel(cl) # Register cluster:
getDoParWorkers() # Find out how many cores are being used
nb <- train(Label ~ .,
            data = train,
            method = "xgbTree")
stopCluster(cl)
registerDoSEQ()
stopCluster(cl)

print(paste("Save XG-Boost", Sys.time()))
save(xgb, file = paste0("results/models/xgboost_sample", SMPL_FRAC*100,".RData"))
rm(xgb)

# KNN --------
print(paste("Training KNN", Sys.time()))
knn <- FNN::knn(train = train[, -1], test = test[, -1], cl = as.factor(train$Label), k=1, algorithm = "kd_tree")
print(paste("Save KNN", Sys.time()))
save(knn, file = paste0("results/models/knn_sample", SMPL_FRAC*100,".RData"))
rm(knn)

# SVM --------
print(paste("Training SVM", Sys.time()))
#svm <- parallelSVM(train[,-1], as.factor(train$Label), numberCores = detectCores(all.tests = FALSE, logical = TRUE))
svm <- LiblineaR(train[,-1], train$Label, type = 2) # type:  1 – L2-regularized L2-loss support vector classification (dual)
print(paste("Save SVM", Sys.time()))
save(svm, file = paste0("results/models/svm_sample", SMPL_FRAC*100,".RData"))
rm(svm)
})
htmlwidgets::saveWidget(p, paste0("results/ProfVis/03_3_Modelling_Final_sample",SMPL_FRAC*100,".html"))
print(paste("Saved ProfVis Analysis!", Sys.time()))

# FINISH ------
print(paste("Finished script!", Sys.time()))
