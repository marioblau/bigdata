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

# Package parallel computing
library(doParallel) #https://www.geeksforgeeks.org/random-forest-with-parallel-computing-in-r-programming/


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
#without parallelization
start.time<-proc.time()
dt <- rpart(Label ~ ., data = train, method = "class")
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
start.time<-proc.time()
dt <- rpart(Label ~ ., data = train, method = "class")
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)
stopCluster(cl)


#save(dt, file = "results/decisiontree.RData")
save(dt, file = "results/scaled/decisiontree.RData")



# RANDOM FOREST ----------
#without parallelization
start.time<-proc.time()
rf <- randomForest(as.factor(Label) ~ ., data = train)
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
start.time<-proc.time()
rf <- randomForest(as.factor(Label) ~ ., data = train)
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)
stopCluster(cl)


#rf <- randomForest(as.factor(Label) ~ ., data = train)
#save(rf, file = "results/randomforest.RData")
save(rf, file = "results/scaled/randomforest.RData")



# XG-BOOST ----------
train_mat <- as.matrix(train)
test_mat <- as.matrix(test)
train_dmat <- xgb.DMatrix(data = train_mat[,-c(1)], label = train_mat[,1])
test_dmat <- xgb.DMatrix(data = test_mat[,-c(1)], label = test_mat[,1])

#without parallelization
start.time<-proc.time()
xgb <- xgboost(data = train_dmat, # the data
               nround = 2, # max number of boosting iterations
               objective = "binary:logistic")  # the objective function
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
start.time<-proc.time()
xgb <- xgboost(data = train_dmat, # the data
               nround = 2, # max number of boosting iterations
               objective = "binary:logistic")  # the objective function
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)
stopCluster(cl)

#save(xgb, file = "results/xgboost.RData")
save(xgb, file = "results/scaled/xgboost.RData")



# LOGISTIC REGRESSION --------
#without parallelization
start.time<-proc.time()
glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
start.time<-proc.time()
glm <- glm(formula = as.factor(Label) ~ ., data = train, family = "binomial")
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)
stopCluster(cl)

save(glm, file = "results/scaled/logregression.RData")



# KNN --------
#without parallelization
start.time<-proc.time()
knn <- knn(train[,-c(1)],test[,-c(1)],cl=train$Label,k=3,prob=TRUE)
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
start.time<-proc.time()
knn <- knn(train[,-c(1)],test[,-c(1)],cl=train$Label,k=3,prob=TRUE)
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)
stopCluster(cl)


save(knn, file = "results/scaled/knn.RData")



# Naive Bayes ------------
#without parallelization
start.time<-proc.time()
nb <- naiveBayes(x = train[,-c(1)], y = train[,c(1)])
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
start.time<-proc.time()
nb <- naiveBayes(x = train[,-c(1)], y = train[,c(1)])
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)
stopCluster(cl)


#nb <- naiveBayes(x = train[,-c(1)], y = train[,c(1)])
save(nb, file = "results/scaled/nb.RData")



# SVM --------
#without parallelization
start.time<-proc.time()
svm <- svm(formula = Label ~ ., data = train[1:10,], #### only 10 rows
           type = 'C-classification',
           kernel = 'linear')
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)

#with parallelization
cl<-makePSOCKcluster(4)
registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
start.time<-proc.time()
svm <- svm(formula = Label ~ ., data = train[1:10,], #### only 10 rows
           type = 'C-classification',
           kernel = 'linear')
stop.time<-proc.time()
run.time<-stop.time -start.time
print(run.time)
stopCluster(cl)


#svm <- svm(formula = Label ~ ., data = train[1:10,], #### only 10 rows
#                 type = 'C-classification',
#                 kernel = 'linear')
save(svm, file = "results/scaled/svm.RData")




