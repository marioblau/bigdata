#-------------------------------------------------------------------------------------
# 03) Results
#-------------------------------------------------------------------------------------
rm(list = ls())
suppressPackageStartupMessages({
  # Models (for prediction)
  library(caret)
  library(randomForest)
  library(ranger)
  library(xgboost)
  library(biglasso)


  # Evaluation
  library(pROC)
  library(grid)
  library(gridExtra)
})
# train test data
load("results/scaled/train_test_sample100_seed2021.RData")

# models
#load("results/scaled/decisiontree_sample100.RData")
#load("results/scaled/randomforest_sample100.RData")
#load("results/scaled/xgboost_sample100.RData")
#load("results/scaled/logregression_sample100.RData")
load("results/scaled/knn_sample100.RData")
#load("results/scaled/nb_sample1.RData")
#load("results/scaled/svm_sample1.RData")


# DECISION TREE --------------
dt.pred <- predict(dt, test, type = 'class')
dt.confm <- confusionMatrix(as.factor(dt.pred), as.factor(test$Label), positive = "1")
save(dt.confm, file = "results/confusionMatrices/decisiontree_sample100")

# RANDOM FOREST ----------
rf.pred <- predict(rf, test, type = "response")
rf.pred2 <- as.factor(max.col(rf.pred$predictions)-1)
rf.confm <- confusionMatrix(rf.pred2, as.factor(test$Label), positive = "1")
save(rf.confm, file = "results/confusionMatrices/randomforest_sample100")

# XG-BOOST ----------
test_mat <- as.matrix(test)
test_dmat <- xgb.DMatrix(data = test_mat[,-1], label = test_mat[, 1])

xgb.pred <- predict(xgb, newdata=test_dmat)
xgb.pred <- round(xgb.pred,0)
xgb.confm <- confusionMatrix(as.factor(xgb.pred), as.factor(test$Label), positive = "1")
save(xgb.confm, file = "results/confusionMatrices/xgboost_sample100")

# LOGISTIC-REGRESSION -------
plot(glm)
coef(glm, lambda = 0.05, drop=TRUE)

ytest_bigmat <- as.big.matrix(test[,-1], type = "double")
glm.pred <- predict(glm, ytest_bigmat, type = "response", lambda = 0.01, drop=TRUE)
glm.pred <- as.matrix(glm.pred)
glm.pred <- round(glm.pred,0)
glm.confm <- confusionMatrix(as.factor(glm.pred), as.factor(test$Label), positive = '1')
save(glm.confm, file = "results/confusionMatrices/logregression_sample100")

# KNN ------- #TODO predict in batches
batches <- split(test[,-1], (seq(nrow(test))-1) %/% 200)
knn.pred <- list()
for(i in seq_along(batches)){
  print(paste0(i,"/", length(batches), " ", Sys.time()))

  batch.pred <- predict(knn, batches[[i]])
  knn.pred <- append(knn.pred, batch.pred)
  print(batch.pred)
}


knn.confm <- confusionMatrix(as.factor(knn.pred), as.factor(test$Label), positive = '1')

# Naive Bayes ----
nb.pred <- predict(nb, newdata = test[,-1])
nb.confm <- confusionMatrix(as.factor(nb.pred), as.factor(test$Label), positive = '1')

# SVM -------
svm.pred <- predict(svm, newdata = test[,-1])
svm.confm <- confusionMatrix(as.factor(svm.pred), as.factor(test$Label), positive = '1')




# ROC / AUC ----------
par(pty = "s")
dt.roc <- roc(test$Label, as.integer(dt.pred), main = "ROC: Decision Tree", col = "#377eb8", lwd = 4, print.auc = TRUE)
xgb.roc <- roc(test$Label, as.integer(xgb.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)
knn.roc <- roc(test$Label, as.integer(knn.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)
glm.roc <- roc(test$Label, as.integer(glm.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)
svm.roc <- roc(test$Label, as.integer(svm.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)
nb.roc <- roc(test$Label, as.integer(nb.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)

dt.roc <- pROC::ggroc(dt.roc, alpha = 0.5, colour = "red", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
xgb.roc <- pROC::ggroc(xgb.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
knn.roc <- pROC::ggroc(knn.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
glm.roc <- pROC::ggroc(glm.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
glm.roc <- pROC::ggroc(svm.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
nb.roc <- pROC::ggroc(nb.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()




g <- arrangeGrob(grobs = list(dt.roc,xgb.roc,knn.roc,glm.roc,glm.roc,nb.roc), ncol = 2, top='Receiver Operator Characteristics')
ggsave(g, file="ROC.png", width = 10, height = 7)
