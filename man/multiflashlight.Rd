% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/multiflashlight.R
\name{multiflashlight}
\alias{multiflashlight}
\alias{multiflashlight.default}
\alias{multiflashlight.flashlight}
\alias{multiflashlight.list}
\alias{multiflashlight.multiflashlight}
\title{Create or Update a multiflashlight}
\usage{
multiflashlight(x, ...)

\method{multiflashlight}{default}(x, ...)

\method{multiflashlight}{flashlight}(x, ...)

\method{multiflashlight}{list}(x, ...)

\method{multiflashlight}{multiflashlight}(x, ...)
}
\arguments{
\item{x}{An object of class "multiflashlight", "flashlight" or a list of flashlights.}

\item{...}{Optional arguments in the flashlights to update, see examples.}
}
\value{
An object of class "multiflashlight" (a named list of flashlight objects).
}
\description{
Combines a list of flashlights to an object of class "multiflashlight"
and/or updates a multiflashlight.
}
\section{Methods (by class)}{
\itemize{
\item \code{multiflashlight(default)}: Used to create a flashlight object.
No \code{x} has to be passed in this case.

\item \code{multiflashlight(flashlight)}: Updates an existing flashlight object and turns
into a multiflashlight.

\item \code{multiflashlight(list)}: Creates (and updates) a multiflashlight from a list
of flashlights.

\item \code{multiflashlight(multiflashlight)}: Updates an object of class "multiflashlight".

}}
\examples{
fit_lm <- lm(Sepal.Length ~ ., data = iris)
fit_glm <- glm(Sepal.Length ~ ., family = Gamma(link = log), data = iris)
mod_lm <- flashlight(model = fit_lm, label = "lm")
mod_glm <- flashlight(model = fit_glm, label = "glm")
(mods <- multiflashlight(list(mod_lm, mod_glm)))
}
\seealso{
\code{\link[=flashlight]{flashlight()}}
}
