## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning = FALSE,
  message = FALSE,
  fig.width = 7,
  fig.height = 6
)

## ----setup--------------------------------------------------------------------
library(dplyr)
library(MetricsWeighted)
library(flashlight)
library(caret)
library(mlr3)
library(mlr3learners)

## -----------------------------------------------------------------------------
data(cars)
str(cars)

## -----------------------------------------------------------------------------
undo_dummies <- function(df, cols) {
  factor(data.matrix(df[, cols]) %*% seq_along(cols), labels = cols)
}

no_yes <- function(x) {
  factor(x, 0:1, c("no", "yes"))
}

# Prepare data
cars <- cars %>% 
  mutate(Price = log(Price),
         Mileage = log(Mileage),
         Made = undo_dummies(., c("Buick", "Cadillac", "Chevy", "Pontiac", "Saab", "Saturn"))) %>% 
  mutate_at(c("Cruise", "Sound", "Leather"), no_yes)

# Response and covariables
y <- "Price"
x <- c("Cylinder", "Doors", "Cruise", "Sound", "Leather", "Mileage", "Made")

# Data split
set.seed(1)
idx <- c(createDataPartition(cars[[y]], p = 0.7, list = FALSE))
tr <- cars[idx, c(y, x)]
te <- cars[-idx, c(y, x)]

# Set up task
task_cars <- TaskRegr$new(id = "cars", backend = tr, target = y)

# Fit the models
fit_lm <- lrn("regr.lm")
fit_lm$train(task_cars)

fit_rf <- lrn("regr.ranger", num.trees = 250, respect.unordered.factors = "partition")
fit_rf$train(task_cars)

## -----------------------------------------------------------------------------
fl_lm <- flashlight(model = fit_lm, label = "lm")
fl_rf <- flashlight(model = fit_rf, label = "rf")

fls <- multiflashlight(list(fl_lm, fl_rf), y = y, data = te, 
                       metrics = list(RMSE = rmse, `R-Squared` = r_squared))

## -----------------------------------------------------------------------------
light_performance(fls) %>% 
  plot(fill = "darkred")

## -----------------------------------------------------------------------------
imp <- light_importance(fls) 
plot(imp, fill = "darkred")

## -----------------------------------------------------------------------------
# Individual conditional expectations (ICE). Using a seed guarantees the same observations across models
light_ice(fls, v = "Cylinder", n_max = 100, seed = 54) %>% 
  plot(alpha = 0.1)

# Partial dependence profiles
light_profile(fls, v = "Cylinder") %>% 
  plot()

light_profile(fls, v = "Cylinder", by = "Leather") %>% 
  plot()

# Accumulated local effects
light_profile(fls, v = "Cylinder", type = "ale") %>% 
  plot()

# M-Plots
light_profile(fls, v = "Mileage", type = "predicted") %>% 
  plot()

# Response profiles, prediction profiles, partial dependence in one
eff <- light_effects(fls, v = "Cylinder") 
eff %>% 
  plot() %>% 
  plot_counts(eff, alpha = 0.3)

## -----------------------------------------------------------------------------
light_interaction(fls, v = most_important(imp, 4), pairwise = TRUE, n_max = 50, seed = 63) %>%
  plot(fill = "darkred")

## -----------------------------------------------------------------------------
light_breakdown(fls, new_obs = te[1, ]) %>% 
  plot(size = 3, facet_ncol = 2)

## -----------------------------------------------------------------------------
light_global_surrogate(fls$rf) %>% 
  plot()

