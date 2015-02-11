\title{rlm}
\name{rlm}
\alias{rlm}
\description{
    Fit a linear model by robust regression taking response in matrix form.
}
\usage{
rlm(formula, weights, acc = 0.01)
}
\arguments{
  \item{formula}{
    a formula of the form \code{y ~ x1 + x2 + \dots}.
}
  \item{weights}{
    a vector of prior weights for each case.
}

  \item{acc}{
    the accuracy for the stopping criterion.
}
}
\details{
  Fitting is done by iterated re-weighted least squares (IWLS).
}
\value{
  matrix of coefficients
}
\examples{
y <- matrix( c(2, 4, 3, 1, 5, 7), nrow=3,  ncol=2) 
x <- c(1, 2, 3)
w <- c(0.9, 0.8, 0.7)
rlm::rlm(y~x, weights = w)
}

