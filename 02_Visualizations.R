#-------------------------------------------------------------------------------------
# 02) Visualizations
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
suppressPackageStartupMessages({
  library(tidyverse)
  library(pryr)
  library(data.table)
  library(caret)
  library(Metrics)
  library(vip)
})

SMPL_FRAC <- 0.01 # 0.01, 0.1, 0.2, ... 0.9, 1.0

# LOAD DATA ----------------
# train test data
load("data/train_test_sample100_seed2021.RData")


# RUNTIME BY %-Data ----------------
runtimes <- read.csv("results/runtimes.csv")

runtimes_pre <- runtimes %>% filter(scriptname == "01_Preprocessing.R")
p <- ggplot(runtimes_pre, aes(x=sample, y=runtime/60))+
  geom_point()+
  geom_smooth(method = "lm", se=FALSE, color = "black", formula = y~x)+
  ylab("Runtime (min)")+
  xlab("% of Data")+
  theme_minimal(base_size = 10)
p
ggsave(p, filename = "results/plots/runtime_preprocessing.png", width = 5, height = 5)


# ACCURACY FINAL MODELS ----------------
load("results/predictions/logregression_pred_sample100.RData")
load("results/predictions/naivebayes_sample100.RData")
load("results/predictions/decisiontree_pred_sample100.RData")
load("results/predictions/randomforest_pred_sample100.RData")
load("results/predictions/xgboost_pred_sample100.RData")
load("results/predictions/knn_pred_sample100.RData")
load("results/predictions/svm_pred_sample50.RData")

pred_list <- list(NaiveBayes = nb.pred,
                  LogisticRegression = glm.pred,
                  DecisionTree = dt.pred,
                  RandomForest = rf.pred,
                  XGBoost = xgb.pred,
                  KNN = knn.pred,
                  SupportVectorMachine = svm.pred)
sapply(pred_list, class)
confm_list <- lapply(pred_list, function(x) confusionMatrix(data = x, reference = as.factor(test$Label), positive = "1"))
acc_list <- lapply(pred_list, function (x) accuracy(test$Label, x))

acc_df <- as.data.frame(acc_list)
data <- tibble(models = names(acc_df), accuracy = t(acc_df)) %>%
    arrange(.,by=desc(accuracy))

p <- ggplot(data, aes(x=accuracy, y=reorder(models, accuracy), fill=as.factor(accuracy)))+
  geom_bar(stat='identity', width = 0.2)+
  scale_fill_brewer(palette = "RdYlGn")+
  coord_cartesian(xlim = c(0, 1))+
  ggtitle("Accuracy of final models")+
  ylab("Models")+
  theme_minimal()+
  theme(legend.position = "none")
p
ggsave(p, filename = "results/plots/accuracy.png", width = 5, height = 5)

# CONFUSION MATRIX ----------------
confm_list$XGBoost
confm_list$RandomForest

# Feature importante RF ----------------
# load models
load("results/models/randomforest_sample1.RData")
load("results/models/xgboost_sample100.RData")

pal <- palette.colors(2, palette = "Okabe-Ito") # colorblind friendly palette

p <- vip::vip(rf, num_features = 15, mapping = aes_string(fill = "Importance"))+
  ggtitle("Feature Importance: Random Forest")+
  xlab("Features")+
  theme_minimal()+
  theme(legend.position = "none")
p
ggsave(p, filename = "results/plots/featureimportance.png", width = 5, height = 5)
