#-------------------------------------------------------------------------------------
# 01) Data Preprocessing
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())

suppressPackageStartupMessages({
  library(tidyverse)
  library(pryr)
  library(data.table)
  library(lubridate)

# Packages runtime analysis
  library(profvis)
  library(htmlwidgets)  # requires to install pandoc! --> https://pandoc.org/installing.html
})

DATA_PATH <- "data/final_dataset.csv"  # original balanced

# TODO select sample size
SMPL_FRAC <- 0.1 # 0.01, 0.1, 0.2, ... 0.9, 1.0

# set save path for final data automatically
SAVE_PATH <- paste0("data/final_dataset_preprocessed_sample",SMPL_FRAC*100,".csv") # sample
SAVE_PATH_SCALED <- paste0("data/final_dataset_preprocessed_sample",SMPL_FRAC*100,"_scaled.csv") # sample

print(paste0("Start Preprocessing on ", SMPL_FRAC*100, "% of data ", Sys.time()))
start.time <- Sys.time()

p <- profvis({ # sample rate = 10ms
  # LOAD DATA ----------------
  print(paste("Loading Data", Sys.time()))

  df_all <- fread(DATA_PATH)
  df <- df_all %>%
    sample_frac(.,SMPL_FRAC) # sample only if SAMPLE variable is true
  rm(df_all) # remove large df from memory

  # REMOVE SPACE FROM COLNAMES ----------------
  print(paste("Clean Colnames", Sys.time()))

  colnames(df) <- gsub(" ", "_", colnames(df))
  colnames(df) <- gsub("/", "_", colnames(df))


  # REMOVE COLUMNS WITHOUT VARIANTION --------
  print(paste("Remove Constant Columns ", Sys.time()))

  df <- df %>% select(-c("Fwd_Byts_b_Avg", "Fwd_Pkts_b_Avg", "Fwd_Blk_Rate_Avg", "Fwd_URG_Flags",
                         "Bwd_Byts_b_Avg", "Bwd_Pkts_b_Avg", "Bwd_Blk_Rate_Avg", "Bwd_URG_Flags"))

  # CONVERT CHARACTER VECTORS TO NUMERIC --------
  print(paste("Clean Data Types", Sys.time()))

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
  print(paste("Remove NA and Inf Values", Sys.time()))

  df_temp <- df %>% drop_na()
  df_temp <- df_temp %>% filter_all(all_vars(!is.infinite(.)))
  (nrow(df) - nrow(df_temp)) / nrow(df) # 476 missing values removed ~0.3 %

  df <- df_temp # replace original df!

  #df %>% sapply(., is.finite) %>% colSums() == nrow(df) # are all columns finite?

  # SAVE ------------
  print(paste("Save Data", Sys.time()))

  fwrite(df, SAVE_PATH)

  # SCALING ------------
  print(paste("Scale Data", Sys.time()))
  df_scaled <- df
  df_scaled <- as.data.frame(scale(df_scaled))
  df_scaled[, 1] <- df[, 1]

  # SAVE ------------
  print(paste("Save Scaled Data", Sys.time()))
  fwrite(df_scaled, SAVE_PATH_SCALED)
})
htmlwidgets::saveWidget(p, paste0("results/ProfVis/01_Preprocessing_sample",SMPL_FRAC*100,".html"))
print(paste("Saved ProfVis Analysis!", Sys.time()))

end.time <- Sys.time()

# save sample and runtime
runtime_data <- tibble(date = Sys.Date(),
                       scriptname = "01_Preprocessing.R",
                       sample = SMPL_FRAC,
                       runtime = round(end.time - start.time,2))

write.table( runtime_data,
             file="results/runtimes.csv",
             append = T,
             sep=',',
             row.names=F,
             col.names=F)



