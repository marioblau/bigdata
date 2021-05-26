#-------------------------------------------------------------------------------------
# 03) Modelling (sequential vs. parallel)
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
suppressPackageStartupMessages({
  # Packages ML models
  library(rpart)
  library(caret)
  library(parallel)
  library(doParallel) # https://www.geeksforgeeks.org/random-forest-with-parallel-computing-in-r-programming/

  # Packages runtime analysis
  library(rbenchmark)
  library(profvis)
  library(htmlwidgets)
  library(microbenchmark)
})

# LOAD DATA ----------------
load("data/train_test_sample1.RData")
#load("data/train_test_sample10.RData")

# NEW APPROACH with PROFVIS----------------
mbm <- microbenchmark(
  "sequential" = {
    dt <- train(Label ~ ., data = train, method = "ctree")
  },
  "parallel" = {
    ncores <- detectCores() #12
    cl <- makeCluster(ncores) # Create cluster with desired number of cores:
    registerDoParallel(cl) # Register cluster:
    getDoParWorkers() # Find out how many cores are being used
    dt <- train(Label ~ ., data = train, method = "ctree")
    stopCluster(cl)
    registerDoSEQ()
  })


# NEW APPROACH with PROFVIS----------------
train$Label <- as.factor(train$Label)

p <- profvis({ # sample rate = 10ms
  # SEQUENTIAL
  dt <- train(Label ~ ., data = train, method = "ctree")
})
htmlwidgets::saveWidget(p, paste0("results/ProfVis/03_3_Decisiontree_SEQUENTIAL.html"))
print("finished")
p <- profvis({ # sample rate = 10ms
  # PARALLEL
  ncores <- detectCores() #12
  cl <- makeCluster(ncores) # Create cluster with desired number of cores:
  registerDoParallel(cl) # Register cluster:
  getDoParWorkers() # Find out how many cores are being used
  dt <- train(Label ~ .,
              data = train,
              method = "ctree")
  stopCluster(cl)
  registerDoSEQ()
})
htmlwidgets::saveWidget(p, paste0("results/ProfVis/03_3_Decisiontree_PARALLEL.html"))










