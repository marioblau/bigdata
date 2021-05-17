#-------------------------------------------------------------------------------------
# 03) Results
#-------------------------------------------------------------------------------------

# Evaluation
library(pROC)
library(grid)
library(gridExtra)
library(caret)

load("results/scaled/train_test_sample1.RData")
load("results/scaled/decisiontree_sample1.RData")
load("results/scaled/knn_sample1.RData")
load("results/scaled/logregression_sample1.RData")
load("results/scaled/nb_sample1.RData")
load("results/scaled/svm_sample1.RData")
load("results/scaled/xgboost_sample1.RData")
load("results/scaled/randomforest_sample1.RData") #does not work


# DECISION TREE --------------
dt.pred <- predict(dt, test, type = 'class')
dt.confm <- confusionMatrix(as.factor(dt.pred), as.factor(test$Label), positive = "1")

# RANDOM FOREST ----------
rf.pred <- predict(rf, newdata=test)
rf.confm <- confusionMatrix(as.factor(rf.pred), as.factor(test$Label), positive = "1")

# XG-BOOST ----------
train_mat <- as.matrix(train)
test_mat <- as.matrix(test)
train_dmat <- xgb.DMatrix(data = train_mat[,-c(1)], label = train_mat[,c(1)])
test_dmat <- xgb.DMatrix(data = test_mat[,-c(1)], label = test_mat[,c(1)])

xgb.pred <- predict(xgb, newdata=test_dmat)
xgb.pred <- round(xgb.pred,0)
xgb.confm <- confusionMatrix(as.factor(xgb.pred), as.factor(test$Label), positive = "1")

# LOGISTIC-REGRESSION -------
glm.pred <- predict(glm, type = "response", newdata = test)
glm.pred <- ifelse(glm.pred < 0.5, 0, 1)
glm.confm <- confusionMatrix(as.factor(glm.pred), as.factor(test$Label), positive = '1')

# KNN -------
knn.pred <- knn
knn.confm <- confusionMatrix(as.factor(knn.pred), as.factor(test$Label), positive = '1')

# SVM -------
svm.pred <- predict(svm, newdata = test[,-c(1)])
svm.confm <- confusionMatrix(as.factor(svm.pred), as.factor(test$Label), positive = '1')

# Naive Bayes ----
nb.pred <- predict(nb, newdata = test[,-c(1)])
nb.confm <- confusionMatrix(as.factor(nb.pred), as.factor(test$Label), positive = '1')


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
