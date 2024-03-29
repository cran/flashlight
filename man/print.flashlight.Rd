% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/print_flashlight.R
\name{print.flashlight}
\alias{print.flashlight}
\title{Prints a flashlight}
\usage{
\method{print}{flashlight}(x, ...)
}
\arguments{
\item{x}{A on object of class "flashlight".}

\item{...}{Further arguments passed from other methods.}
}
\value{
Invisibly, the input is returned.
}
\description{
Print method for an object of class "flashlight".
}
\examples{
fit <- lm(Sepal.Length ~ ., data = iris)
x <- flashlight(model = fit, label = "lm", y = "Sepal.Length", data = iris)
x
}
\seealso{
\code{\link[=flashlight]{flashlight()}}
}
