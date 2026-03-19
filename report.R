### ASML Classification Summative ###
#####################################

#### install dependencies ####
# install.packages()

#### load data ####
heartfailure <- read.csv("https://www.maths.dur.ac.uk/users/hailiang.du/assignment_data/heart_failure.csv", header=TRUE)
head(heartfailure)


#### EDA ####
dim(heartfailure)
# only 299 observations with 13 variables 

library(skimr)
skim(heartfailure)

# no missing values 
DataExplorer::plot_bar(heartfailure, by = "fatal_mi")
# DataExplorer::plot_histogram(heartfailure)
DataExplorer::plot_boxplot(heartfailure, by = "fatal_mi", ncol = 3) # alternative to hist for numeric

library(dplyr)
# library(ggplot2)
# ggplot(heartfailure, aes(x=time, fill=factor(fatal_mi))) +
#   geom_histogram()
# 
# pairs(heartfailure |> select(age, creatinine_phosphokinase, 
#                              ejection_fraction, platelets, serum_creatinine, serum_sodium, time))
# 
# library(GGally)
# ggpairs(heartfailure |> select(fatal_mi, age, creatinine_phosphokinase, 
#                              ejection_fraction, platelets, serum_creatinine, serum_sodium, time),
#       aes(color=factor(fatal_mi)))
# 


#### model fitting ####
set.seed(123)

### preprocessing 
library("rsample")
# ensure fatal_mi is a factor 
heartfailure$fatal_mi <- factor(heartfailure$fatal_mi)

# use train-validate-test 70-15-15
heart_split <- initial_split(heartfailure, prop=0.70)
heart_train_no_preprocessing <- training(heart_split)
heart_test_no_preprocessing <- testing(heart_split)



### models

library("mlr3verse")
# mlr_learners

heart_task <- TaskClassif$new(id = "HeartFatality",
                              backend = heart_train_no_preprocessing,  # exclude test data during model selection
                              target = "fatal_mi",
                              positive = '1')

cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(heart_task) # fixed folds for fair comparison between multiple learners

library("xgboost")
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart <- lrn("classif.rpart", predict_type = "prob")  # prune this?
lrn_forest <- lrn("classif.ranger", predict_type = "prob")
lrn_boost <- lrn("classif.xgboost", predict_type = "prob")

# benchmark default learners 
res <- benchmark(data.table(
  task       = list(heart_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    lrn_forest,
                    lrn_boost),
  resampling = list(cv5)
), store_models = TRUE)

# metrics for evaluation
res$aggregate(list(msr("classif.acc"),
                   msr("classif.fnr"),  # small FNR same as high recall 
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.ce")))


# predict on training data and compare metric to cv metrics 
# lrn_forest$train(heart_task)
# pred <- lrn_forest$predict(heart_task)
# pred$score(list(msr("classif.acc"),
#                 msr("classif.fnr"),  # small FNR same as high recall 
#                 msr("classif.auc"),
#                 msr("classif.fpr"),
#                 msr("classif.ce")))
# pred$confusion

# random forest appears best 

# view hyperparameters that could be tuned
### tuning rarely needed with random forest 
# lrn_forest$param_set
# max_depth, min_child_weight, num.trees, regularization.factor 
# library(mlr3tuning)
# 
# lrn_forest_tuned <- lrn("classif.ranger", 
#                         ...
#                         predict_type = "prob")
# 
# instance = ti(
#   task = list(heart_task),
#   learner = lrn_forest_tuned,
#   resampling = list(cv5),
#   measures = msr("classif.ce"),
#   terminator = trm("none")
# )
# 
# tuner = tnr("grid_search", resolution = 5)
# tuner$optimize(instance)


### deep learning model 
# keras requires that all data is numeric -> factors are handled in preprocessing below
# heartfailure$fatal_mi <- as.numeric(as.character(heartfailure$fatal_mi))
heartfailure$fatal_mi <- as.factor(heartfailure$fatal_mi)
heartfailure$anaemia <- as.factor(heartfailure$anaemia)
heartfailure$diabetes <- as.factor(heartfailure$diabetes)
heartfailure$high_blood_pressure <- as.factor(heartfailure$high_blood_pressure)
heartfailure$sex <- as.factor(heartfailure$sex)
heartfailure$smoking <- as.factor(heartfailure$smoking)

## prepare data for deep learning model
# train/validate/test split
library("rsample")
# do initial_split, then use training() and testing() to extract relevant splits of data
heart_split <- initial_split(heartfailure, prop=0.7)
heart_split2 <- initial_split(testing(heart_split), 0.5)
heart_train <- training(heart_split)
heart_validate <- training(heart_split2)
heart_test <- testing(heart_split2)

# preprocessing 
library(recipes)
cake <- recipe(fatal_mi ~ ., data = heartfailure) |>
  step_center(all_numeric()) |> # center by subtracting the mean from all numeric features
  step_scale(all_numeric()) |> # scale by dividing by the standard deviation on all numeric features
  step_dummy(all_nominal(), one_hot = TRUE) |> # turn all factors into a one-hot coding
  prep(training = heart_train) # learn all the parameters of preprocessing on the training data

# apply preprocessing (based on parameters from training data)
# ie don't learn new preprocessing params for the test data
heart_train_final <- bake(cake, new_data = heart_train)
heart_validate_final <- bake(cake, new_data = heart_validate)
heart_test_final <- bake(cake, new_data = heart_test)

# split into X and y
heart_train_X <- heart_train_final |> 
  select(-starts_with("fatal_mi_")) |>
  as.matrix()
heart_train_y <- heart_train_final |> 
  select(fatal_mi_X1) |>
  as.matrix()

heart_validate_X <- heart_validate_final |> 
  select(-starts_with("fatal_mi_")) |>
  as.matrix()
heart_validate_y <- heart_validate_final |> 
  select(fatal_mi_X1) |>
  as.matrix()

heart_test_X <- heart_test_final |> 
  select(-starts_with("fatal_mi_")) |>
  as.matrix()
heart_test_y <- heart_test_final |> 
  select(fatal_mi_X1) |>
  as.matrix()

dim(heart_validate_X)
head(heart_test_X)
dim(heartfailure)

# deep NN architecture 
library(keras3)

deep.net1 <- keras_model_sequential() |>
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(heart_train_X))) |>
  layer_dense(units = 32, activation = "relu") |>
  layer_dense(units = 1, activation = "sigmoid")

# add dropout to reduce overfitting
deep.net2 <- keras_model_sequential() |>
  layer_dense(units = 32, activation = "relu",
              input_shape = c(ncol(heart_train_X))) |>
  layer_dropout(rate = 0.2, seed=123) |>
  layer_dense(units = 32, activation = "relu") |>
  layer_dropout(rate = 0.4, seed=123) |>
  layer_dense(units = 1, activation = "sigmoid")

deep.net3 <- keras_model_sequential() |>
  layer_dense(units = 64, activation = "relu",
              input_shape = c(ncol(heart_train_X))) |>
  layer_dropout(rate = 0.3, seed=123) |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dropout(rate = 0.5, seed=123) |>
  layer_dense(units = 1, activation = "sigmoid")

### deep learning model 1
# compile the neural network
deep.net1 |> compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),  # which optimiser to choose? rmsprop and adam pretty similar performance
  metrics = c("accuracy", "Recall") # want high recall = 1 - FNR (small FNR)
)

# fit the neural network 
# using mini batches heuristic choice of size 32
deep.net1 |> fit(
  heart_train_X, heart_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(heart_validate_X, heart_validate_y),
)

### model 2
# compile the neural network
deep.net2 |> compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),  # which optimiser to choose? rmsprop and adam pretty similar performance
  metrics = c("accuracy", "Recall") # want high recall = 1 - FNR (small FNR)
)

# fit the neural network 
# using mini batches heuristic choice of size 32
deep.net2 |> fit(
  heart_train_X, heart_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(heart_validate_X, heart_validate_y),
)


# compile the neural network
deep.net3 |> compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),  # which optimiser to choose? rmsprop and adam pretty similar performance
  metrics = c("accuracy", "Recall") # want high recall = 1 - FNR (small FNR)
)

# fit the neural network 
# using mini batches heuristic choice of size 32
deep.net3 |> fit(
  heart_train_X, heart_train_y,
  epochs = 50, batch_size = 32,
  validation_data = list(heart_validate_X, heart_validate_y),
)

#####################################################################################
# To get the probability predictions on the test set:
# pred_test_prob <- deep.net |> predict(heart_test_X)

# To get the raw classes (assuming 0.5 cutoff):
# pred_test_res <- deep.net |> predict(heart_test_X) |> (`>`)(0.5) |> as.integer()

# Confusion matrix/accuracy/AUC metrics
# table(pred_test_res, heart_test_y)
# yardstick::accuracy_vec(as.factor(heart_test_y),
#                         as.factor(pred_test_res))
# yardstick::roc_auc_vec(factor(heart_test_y, levels = c("1","0")),
#                        c(pred_test_prob))
######################################################################################


#### Final model performance ####

lrn_forest$train(heart_task)
heart_test_task <- TaskClassif$new("test", backend = heart_test_no_preprocessing, target = "fatal_mi", positive = '1')
pred <- lrn_forest$predict(heart_test_task)
pred$score(list(msr("classif.acc"),
                msr("classif.fnr"),  # small FNR same as high recall 
                msr("classif.auc"),
                msr("classif.fpr"),
                msr("classif.ce")))
pred$confusion




