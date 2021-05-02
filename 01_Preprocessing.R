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
SAVE_PATH_SMPL <- "data/final_dataset_preprocessed_sample.csv"  #

# LOAD DATA ----------------
df_all <- fread(DATA_PATH)
df <- df_all %>% sample_frac(0.01) #TODO select SAMPLE (y/n)
str(df)

# REMOVE SPACE FROM COLNAMES ----------------
colnames(df) <- gsub(" ", "_", colnames(df))
colnames(df) <- gsub("/", "_", colnames(df))


# REMOVE COLUMNS WITHOUT VARIANTION --------
df <- df %>% select(-c("Fwd_Byts_b_Avg", "Fwd_Pkts_b_Avg", "Fwd_Blk_Rate_Avg", "Fwd_URG_Flags",
                       "Bwd_Byts_b_Avg", "Bwd_Pkts_b_Avg", "Bwd_Blk_Rate_Avg", "Bwd_URG_Flags"))


# CONVERT CHARACTER VECTORS TO NUMERIC --------
#df %>% sapply(., class)  # Character vectors: Flow ID, Src IP, Dst IP, Timestamp

# Convert dates
df <- df %>%
  mutate(Timestamp = lubridate::dmy_hms(Timestamp), # convert character to date
         year = year(Timestamp),
         month = month(Timestamp),
         day = day(Timestamp),
         hour = hour(Timestamp)) %>%
  select(-Timestamp) # drop timestamp column

# Convert Label (ddos = 1, benign = 0)
df <- df %>%
  mutate(Label = as.integer(as.factor(Label)) -1)  # -1 to convert labels from (1,2) to (0,1)

# Convert Flow ID
df <- df %>%
  separate(Flow_ID, c("FlowID1", NA, NA, NA, NA), sep = "[-]") %>%
  mutate(FlowID1 = if_else(FlowID1 == Src_IP, 1, 0))

# Convert ScrIP
df <- df %>%
  separate(Src_IP, c("ScrIP1", "ScrIP2", "ScrIP3", "ScrIP4"), sep = "[.]") %>%
  separate(Dst_IP, c("DstIP1", "DstIP2", "DstIP3", "DstIP4"), sep = "[.]") %>%
  mutate(
    ScrIP1 = as.integer(ScrIP1),
    ScrIP2 = as.integer(ScrIP2),
    ScrIP3 = as.integer(ScrIP3),
    ScrIP4 = as.integer(ScrIP4),
    DstIP1 = as.integer(DstIP1),
    DstIP2 = as.integer(DstIP2),
    DstIP3 = as.integer(DstIP3),
    DstIP4 = as.integer(DstIP4)
  )

# Put Label at first position
df <- df %>%
  select(Label, everything())

# REMOVE NA / INF VALUES --------------
df_temp <- df %>% drop_na()
df_temp <- df_temp %>% filter_all(all_vars(!is.infinite(.)))
(nrow(df) - nrow(df_temp)) / nrow(df) # 476 missing values removed ~0.3 %

df <- df_temp # replace original df!

#df %>% sapply(., is.finite) %>% colSums() == nrow(df) # are all columns finite?

# SAVE ------------
fwrite(df, SAVE_PATH_SMPL) #TODO select SAMPLE (y/n)