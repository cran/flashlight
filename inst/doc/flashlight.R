## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning = FALSE,
  message = FALSE,
  fig.width = 5.5,
  fig.height = 4.5
)

## -----------------------------------------------------------------------------
library(ggplot2)
library(MetricsWeighted)
library(flashlight)

fit_lm <- lm(Sepal.Length ~ ., data = iris)

# Make explainer object
fl_lm <- flashlight(
  model = fit_lm, 
  data = iris, 
  y = "Sepal.Length", 
  label = "lm",               
  metrics = list(RMSE = rmse, `R-squared` = r_squared)
)

## -----------------------------------------------------------------------------
fl_lm |> 
  light_performance() |> 
  plot(fill = "darkred") +
  labs(x = element_blank(), title = "Performance on training data")

fl_lm |> 
  light_performance(by = "Species") |> 
  plot(fill = "darkred") +
  ggtitle("Performance split by Species")

## -----------------------------------------------------------------------------
fl_lm |>
  light_importance(m_repetitions = 4) |> 
  plot(fill = "darkred") +
  labs(title = "Permutation importance", y = "Increase in RMSE")

## -----------------------------------------------------------------------------
fl_lm |> 
  light_ice("Sepal.Width", n_max = 200) |> 
  plot(alpha = 0.3, color = "chartreuse4") +
  labs(title = "ICE curves for 'Sepal.Width'", y = "Prediction")

fl_lm |> 
  light_ice("Sepal.Width", n_max = 200, center = "middle") |> 
  plot(alpha = 0.3, color = "chartreuse4") +
  labs(title = "c-ICE curves for 'Sepal.Width'", y = "Prediction (centered)")

## -----------------------------------------------------------------------------
fl_lm |> 
  light_profile("Sepal.Width", n_bins = 40) |> 
  plot() +
  ggtitle("PDP for 'Sepal.Width'")

fl_lm |> 
  light_profile("Sepal.Width", n_bins = 40, by = "Species") |> 
  plot() +
  ggtitle("Same grouped by 'Species'")

## -----------------------------------------------------------------------------
fl_lm |> 
  light_profile2d(c("Petal.Width", "Petal.Length")) |> 
  plot()

## -----------------------------------------------------------------------------
fl_lm |> 
  light_profile("Sepal.Width", type = "ale") |> 
  plot() +
  ggtitle("ALE plot for 'Sepal.Width'")

## -----------------------------------------------------------------------------
fl_lm |> 
  light_effects("Sepal.Width") |> 
  plot(use = "all") +
  ggtitle("Different types of profiles for 'Sepal.Width'")

## -----------------------------------------------------------------------------
fl_lm |> 
  light_breakdown(new_obs = iris[1, ]) |> 
  plot()

## -----------------------------------------------------------------------------
fl_lm |> 
  light_global_surrogate() |> 
  plot()

## -----------------------------------------------------------------------------
library(rpart)

fit_tree <- rpart(
  Sepal.Length ~ ., 
  data = iris, 
  control = list(cp = 0, xval = 0, maxdepth = 5)
)

# Make explainer object
fl_tree <- flashlight(
  model = fit_tree, 
  data = iris, 
  y = "Sepal.Length", 
  label = "tree",               
  metrics = list(RMSE = rmse, `R-squared` = r_squared)
)

# Combine with other explainer
fls <- multiflashlight(list(fl_tree, fl_lm))

fls |> 
  light_performance() |> 
  plot(fill = "chartreuse4") +
  labs(x = "Model", title = "Performance")

fls |> 
  light_importance() |> 
  plot(fill = "chartreuse4") +
  labs(y = "Increase in RMSE", title = "Permutation importance")

fls |> 
  light_profile("Petal.Length", n_bins = 40) |> 
  plot() +
  ggtitle("PDP")

fls |> 
  light_profile("Petal.Length", n_bins = 40, by = "Species") |> 
  plot() +
  ggtitle("PDP by Species")

