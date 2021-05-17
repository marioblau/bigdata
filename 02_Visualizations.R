#-------------------------------------------------------------------------------------
# 02) Visualizations
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
suppressPackageStartupMessages({
  library(tidyverse)
  library(pryr)
  library(data.table)
})

SMPL_FRAC <- 0.01 # 0.01, 0.1, 0.2, ... 0.9, 1.0
DATA_PATH <- paste0("data/final_dataset_preprocessed_sample100_scaled.csv")

# LOAD DATA ----------------
print(paste("Loading data from: ",DATA_PATH))
df <- fread(DATA_PATH) %>%
  sample_frac(.,SMPL_FRAC) # sample only if SAMPLE variable is true

# SUMMARY STATS ----------------
# TODO: Compute Summary Stats

# RUNTIME BY %-Data ----------------
runtimes <- read.csv("results/runtimes.csv")

ggplot(runtimes, aes(x=SMPL_FRAC, y=runtimes)) %>%
  geom_point()+
  geom_