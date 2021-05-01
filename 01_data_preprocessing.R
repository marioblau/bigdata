#-------------------------------------------------------------------------------------
# 01) Data Preprocessing
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
library(tidyverse)
library(pryr)
library(data.table)
library(lubridate)

DATA_PATH <- "../bigdata/data/final_dataset.csv"  # original balanced
SAVE_PATH <- "../bigdata/data/final_dataset_preprocessed.csv"  #
SAVE_PATH_SMPL <- "../bigdata/data/final_dataset_preprocessed_sample.csv"  #

# LOAD DATA ----------------
df_all <- fread(DATA_PATH)
df <- df_all %>% sample_frac(0.01) # TODO select SAMPLE

# PREPROCESSING ----------------
# replace colname spaces with _
colnames(df) <- sub(" ", "_", colnames(df))

# character verctors: Flow ID, Src IP, Dst IP, Timestamp
df %>% sapply(., class)  # show all column types

# convert dates
df <- df %>%
  mutate(Timestamp = lubridate::dmy_hms(Timestamp), # convert character to date
         year = year(Timestamp),
         month = month(Timestamp),
         day = day(Timestamp),
         hour = hour(Timestamp)) %>%
  select(-Timestamp) # drop timestamp column

# convert Label to numeric  (ddos = 1, benign = 0)
df <- df %>%
  mutate(Label = as.numeric(as.factor(Label))-1)  # -1 to convert labels from (1,2) to (0,1)

# seperate Flow ID
df <- df %>%
  separate(Flow_ID, c("FlowID1", NA, NA, NA, NA), sep = "[-]") %>%
  mutate(FlowID1 = if_else(FlowID1 == Src_IP, 1, 0))

# seperate ScrIP
df <- df %>%
  separate(Src_IP, c("ScrIP1", "ScrIP2", "ScrIP3", "ScrIP4"), sep = "[.]") %>%
  separate(Dst_IP, c("DstIP1", "DstIP2", "DstIP3", "DstIP4"), sep = "[.]")


# SAVE ------------
fwrite(df, SAVE_PATH_SMPL)