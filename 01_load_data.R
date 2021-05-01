#-------------------------------------------------------------------------------------
# 01) Load Data
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
library(tidyverse)
library(pryr)
library(data.table)

DATA_PATH <- "../bigdata/data/final_dataset.csv"  # original balanced
SAVE_PATH <- "../bigdata/data/final_dataset_preprocessed.csv"  #

# sample data for debugging
SAMPLE <- TRUE
SAMPLE_FRAC <- 0.01


# DATA IMPORT ----------------
mem_change(df <- fread(DATA_PATH))

if(SAMPLE== TRUE){
  mem_change(df_smpl <- df %>% sample_frac(SAMPLE_FRAC))
}

df %>% colnames
# summary(df)

# PREPROCESSING ----------------

# convert Label to numeric  (ddos = 1, benign = 0)
df <- df %>%
  mutate(Label01 = as.numeric(as.factor(Label))-1)  # -1 to convert labels from (1,2) to (0,1)

sum(df$Label01) / length(df$Label01) # ~ 50/50 ddos vs benign
summary(df$Label01)

# SAVE ------------
fwrite(df, SAVE_PATH)