#-------------------------------------------------------------------------------------
# 01) Load Data
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
library(tidyverse)
library(pryr)
library(data.table)

# Packages Trees
library(rpart)

# LOAD DATA ----------------
DATA_PATH <- "../bigdata/data/final_dataset_preprocessed.csv"

# sample data for debugging
SAMPLE <- TRUE
SAMPLE_FRAC <- 0.01

mem_change(df <- fread(DATA_PATH))
if(SAMPLE == TRUE){
  mem_change(df <- df %>% sample_frac(SAMPLE_FRAC))
}

head(df,2)

# TRAIN TEST SPLIT ---------------- #TODO evtl cross validation
df$id <- 1:nrow(df)
train <- df %>% sample_frac(.95)
test  <- anti_join(df, train, by = 'id')

dim(train)

# TRAIN DECISION TREE --------------
dt <- rpart(Label01 ~ ., data = train)

summary(dt)
plot(dt)
text(dt)

df %>% sapply(.,class)


