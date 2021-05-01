#-------------------------------------------------------------------------------------
# 01) Load Data
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
library(tidyverse)
library(pryr)
library(data.table)

# LOAD DATA ----------------
DATA_PATH <- "../bigdata/data/final_dataset_preprocessed.csv"

# sample data for debugging
SAMPLE <- TRUE
SAMPLE_FRAC <- 0.01

# DATA IMPORT ----------------
mem_change(df <- fread(DATA_PATH))





