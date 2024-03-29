.onLoad <- function(libname, pkgname) {
  op <- options()
  op.flashlight <- list(
    flashlight.after_name = "after",
    flashlight.before_name = "before",
    flashlight.counts_name = "counts",
    flashlight.description_name = "description",
    flashlight.error_name = "error",
    flashlight.id_name = "id",
    flashlight.label_name = "label",
    flashlight.metric_name = "metric",
    flashlight.q1_name = "q1",
    flashlight.q3_name = "q3",
    flashlight.step_name = "step",
    flashlight.tree_name = "tree",
    flashlight.type_name = "type",
    flashlight.value_name = "value",
    flashlight.variable_name = "variable"
  )
  toset <- !(names(op.flashlight) %in% names(op))
  if (any(toset)) {
    options(op.flashlight[toset])
  }
  invisible()
}

utils::globalVariables(
  c("fill_", "high_", "lab_", "low_", "temp_", "xmax_", "xmin_", "y_"))

