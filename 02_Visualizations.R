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

runtimes_pre <- runtimes %>% filter(scriptname == "01_Preprocessing.R")
p <- ggplot(runtimes_pre, aes(x=sample, y=runtime/60))+
  geom_point()+
  geom_smooth(method = "lm", se=FALSE, color = "black", formula = y~x)+
  ylab("Runtime (min)")+
  xlab("% of Data")+
  theme_minimal(base_size = 10)
p
ggsave(p, filename = "results/plots/runtime_preprocessing.png", width = 5, height = 5)

