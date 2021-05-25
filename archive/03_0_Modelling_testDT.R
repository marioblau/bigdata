#-------------------------------------------------------------------------------------
# 03) Modelling (sequential vs. parallel)
#-------------------------------------------------------------------------------------
# Packages and settings
rm(list = ls())
suppressPackageStartupMessages({
  # Packages ML models
  library(rpart)
  library(doParallel) # https://www.geeksforgeeks.org/random-forest-with-parallel-computing-in-r-programming/

  # Packages runtime analysis
  library(rbenchmark)
})

# LOAD DATA ----------------
load("data/train_test_sample1.RData")
load("data/train_test_sample10.RData")
benchmark("sequential" = {
            dt <- rpart(Label ~ ., data = train, method = "class")
          },
          "parallel" = {
            cl <- makePSOCKcluster(4)
            registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
            dt <- rpart(Label ~ ., data = train, method = "class")
            stopCluster(cl)
          },
          replications = 10,
          columns = c("test", "replications", "elapsed", "relative", "user.self", "sys.self")
)



# parallel 10%
system.time({
    cl <- makePSOCKcluster(4)
    registerDoParallel(cl, cores = detectCores(all.tests = FALSE, logical = TRUE))
    dt <- rpart(Label ~ ., data = train, method = "class")
    stopCluster(cl)
})
# not parallel 10%
system.time({
    dt <- rpart(Label ~ ., data = train, method = "class")
})