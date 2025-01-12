# Analysis of High-Dimensional Estimation Techniques

## Overview

This project focuses on the analysis of high-dimensional estimation techniques, particularly the inadmissibility of the sample mean in high dimensions and the James-Stein estimator. The project includes graphical illustrations, Monte Carlo risk estimates, and simulations to demonstrate the effectiveness of the James-Stein estimator over the sample mean.

## Contents

- [Introduction](#introduction)
- [Graphical Illustration](#graphical-illustration)
- [Monte Carlo Risk Estimates](#monte-carlo-risk-estimates)
- [Additional Estimators](#additional-estimators)
- [References](#references)

## Introduction

The project explores the concept of shrinking the sample mean towards the origin in high-dimensional spaces to reduce variance. The James-Stein estimator is introduced as a preferred method over the sample mean due to its ability to reduce variance by introducing some bias.

## Graphical Illustration

The graphical illustrations provide a visual understanding of the James-Stein estimator's effect on near-field and far-field points. The illustrations show how the estimator shrinks the sample mean towards the origin, reducing variance while introducing some bias.

*James-Stein Estimator, and its effect on near-field and far-field points.*

## Monte Carlo Risk Estimates

Monte Carlo simulations are used to compare the risk of estimating the true mean using the Maximum Likelihood Estimator (MLE) and the James-Stein Estimator. The results show that the risk of the James-Stein estimator is consistently lower than that of the MLE, especially as the dimensionality increases.

*Monte Carlo Risk Estimates*

## Additional Estimators

The project also explores additional estimators such as $\delta_{JSO}(\boldsymbol{X})$ and $\delta_{JS}^+(\boldsymbol{X})$, which further demonstrate the inadmissibility of the James-Stein estimator and provide even lower risk in certain scenarios.

*Risk of JSO*

*Risk of JS+*

## References

- Wasserman, L. (2004). All of Statistics.
- Wikipedia contributors. (2023, October 10). Chi distribution. In Wikipedia, The Free Encyclopedia. Retrieved from [Chi distribution](https://en.wikipedia.org/wiki/Chi_distribution)

For more details, please refer to the "report/report.pdf". 
