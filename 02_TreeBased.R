#-------------------------------------------------------------------------------------
# 01) Load Data
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
library(tidyverse)
library(pryr)
library(data.table)

DATA_PATH <- "../bigdata/data/unbalaced_20_80_dataset.csv"

# sample data for debugging
SAMPLE <- TRUE
SAMPLE_FRAC <- 0.01


# LOAD DATA ----------------
mem_change(df <- fread(DATA_PATH))

if(SAMPLE== TRUE){
  mem_change(df <- df %>% sample_frac(SAMPLE_FRAC))
  # mem_change(rm(df))
}

gc()

df %>% colnames
summary(df$Label)
class(df)

df$Label %>% head(6)
