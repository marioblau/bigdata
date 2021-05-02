# Title     : TODO
# Objective : TODO
# Created by: sdien
# Created on: 02/05/2021



# DECISION TREE --------------
dt.pred <- predict(dt, test, type = 'class')
dt.confm <- confusionMatrix(as.factor(dt.pred), as.factor(test$Label), positive = "1")

# RANDOM FOREST ----------
rf.pred <- predict(rf, newdata=test)
rf.confm <- confusionMatrix(as.factor(rf.pred), as.factor(test$Label), positive = "1")

# XG-BOOST ----------
train_mat <- as.matrix(train)
test_mat <- as.matrix(test)
train_dmat <- xgb.DMatrix(data = train_mat[,2:95], label = train_mat[,1])
test_dmat <- xgb.DMatrix(data = test_mat[,2:95], label = test_mat[,1])

xgb.pred <- predict(xgb, newdata=test_dmat)
xgb.pred <- round(xgb.pred,0)
xgb.confm <- confusionMatrix(as.factor(xgb.pred), as.factor(test$Label), positive = "1")



# ROC / AUC ----------
par(pty = "s")
dt.roc <- roc(test$Label, as.integer(dt.pred), main = "ROC: Decision Tree", col = "#377eb8", lwd = 4, print.auc = TRUE)
xgb.roc <- roc(test$Label, as.integer(xgb.pred), main = "ROC: XG-Boost", col = "#377eb8", lwd = 4, print.auc = TRUE)

dt.roc <- pROC::ggroc(dt.roc, alpha = 0.5, colour = "red", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()
xgb.roc <- pROC::ggroc(xgb.roc, alpha = 0.5, colour = "blue", linetype = 2, size = 2) + geom_abline(intercept = 1, slope = 1) + theme_minimal()

g <- arrangeGrob(grobs = list(dt.roc,xgb.roc), ncol = 2, top='Receiver Operator Characteristics')
ggsave(g, file="ROC.png", width = 10, height = 7)
