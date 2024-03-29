% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/response.R
\name{response}
\alias{response}
\alias{response.default}
\alias{response.flashlight}
\alias{response.multiflashlight}
\title{Response of multi/-flashlight}
\usage{
response(object, ...)

\method{response}{default}(object, ...)

\method{response}{flashlight}(object, ...)

\method{response}{multiflashlight}(object, ...)
}
\arguments{
\item{object}{An object of class "flashlight".}

\item{...}{Arguments used to update the flashlight before extracting the response.}
}
\value{
A numeric vector of responses.
}
\description{
Extracts response from object of class "flashlight".
}
\section{Methods (by class)}{
\itemize{
\item \code{response(default)}: Default method not implemented yet.

\item \code{response(flashlight)}: Extract response from flashlight object.

\item \code{response(multiflashlight)}: Extract responses from multiflashlight object.

}}
\examples{
fit <- lm(Sepal.Length ~ ., data = iris)
(fl <- flashlight(model = fit, data = iris, y = "Sepal.Length", label = "ols"))
response(fl)[1:5]
response(fl, data = iris[1:5, ])
response(fl, data = iris[1:5, ], linkinv = exp)
}
