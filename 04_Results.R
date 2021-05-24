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
  library(e1071)


  # Evaluation
  library(pROC)
  library(grid)
  library(gridExtra)
})
# train test data
load("data/train_test_sample100_seed2021.RData")

# models
load("results/models/decisiontree_sample100.RData")
load("results/models/randomforest_sample100.RData")
load("results/models/xgboost_sample100.RData")
load("results/models/logregression_sample100.RData")
load("results/models/knn_sample100.RData")
load("results/models/nb_sample100.RData")

load("results/models/svm_sample1.RData")

# PREDICTIONS ----------------------------------------------
# Naive Bayes
nb.pred <- predict(nb, newdata = test[,-1])
save(nb.pred, file = "results/predictions/naivebayes_sample100")

# Logistic Regression
plot(glm)
coef(glm, lambda = 0.05, drop=TRUE)

ytest_bigmat <- as.big.matrix(test[,-1], type = "double")
glm.pred <- predict(glm, ytest_bigmat, type = "response", lambda = 0.01, drop=TRUE)
glm.pred <- as.matrix(glm.pred)
glm.pred <- round(glm.pred,0)
save(glm.pred, file = "results/predictions/logregression_pred_sample100")

# Decision Tree
dt.pred <- predict(dt, test, type = 'class')
dt.pred <- as.factor(dt.pred)
save(dt.pred, file = "results/predictions/decisiontree_pred_sample100")

# Random Forest
rf.pred <- predict(rf, test, type = "response")
rf.pred <- as.factor(max.col(rf.pred$predictions)-1)
save(rf.pred, file = "results/predictions/randomforest_pred_sample100")

# XG-Boost
test_mat <- as.matrix(test)
test_dmat <- xgb.DMatrix(data = test_mat[,-1], label = test_mat[, 1])

xgb.pred <- predict(xgb, newdata=test_dmat)
xgb.pred <- round(xgb.pred,0)
save(xgb.pred, file = "results/predictions/xgboost_pred_sample100")

# KNN
knn.pred <- knn
save(knn.pred, file = "results/predictions/knn_pred_sample100")

# SVM
svm.pred <- predict(svm, newdata = test[,-1])


# Confusion Matrices ----------------------------------------------
nb.confm <- confusionMatrix(as.factor(nb.pred), as.factor(test$Label), positive = '1')
glm.confm <- confusionMatrix(as.factor(glm.pred), as.factor(test$Label), positive = '1')

dt.confm <- confusionMatrix(dt.pred, as.factor(test$Label), positive = "1")
rf.confm <- confusionMatrix(rf.pred2, as.factor(test$Label), positive = "1")
xgb.confm <- confusionMatrix(as.factor(xgb.pred), as.factor(test$Label), positive = "1")

knn.confm <- confusionMatrix(as.factor(knn.pred), as.factor(test$Label), positive = '1')
svm.confm <- confusionMatrix(as.factor(svm.pred), as.factor(test$Label), positive = '1')

# ROC / AUC ----------
par(pty = "s")
nb.roc <- roc(test$Label, as.integer(nb.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)
glm.roc <- roc(test$Label, as.integer(glm.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)
dt.roc <- roc(test$Label, as.integer(dt.pred), main = "ROC: Decision Tree", col = "#377eb8", lwd = 4, print.auc = TRUE)
xgb.roc <- roc(test$Label, as.integer(xgb.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)
knn.roc <- roc(test$Label, as.integer(knn.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)
svm.roc <- roc(test$Label, as.integer(svm.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)

nb.roc <- pROC::ggroc(nb.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
glm.roc <- pROC::ggroc(svm.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
dt.roc <- pROC::ggroc(dt.roc, alpha = 0.5, colour = "red", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
xgb.roc <- pROC::ggroc(xgb.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
knn.roc <- pROC::ggroc(knn.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
glm.roc <- pROC::ggroc(glm.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()




g <- arrangeGrob(grobs = list(dt.roc,xgb.roc,knn.roc,glm.roc,glm.roc,nb.roc), ncol = 2, top='Receiver Operator Characteristics')
ggsave(g, file="ROC.png", width = 10, height = 7)
