# Advanced Regression Techniques for Natural Gas Price Forecasting

## Abstract

This repository hosts an advanced natural gas price forecasting model, which compares the efficacy of various regression techniques. Due to the continuous nature of commodity prices, we employ supervised regression algorithms to predict natural gas prices. This project not only offers a methodical comparison of diverse regression approaches—Linear Regression, Polynomial Regression, Decision Trees, Random Forest, and Artificial Neural Network Regression—but also evaluates their predictive performance in the context of market volatility and data complexity.

## Contributors

- Alberto Biscalchin
- Adnane Soulaimani

## Methodology

The project meticulously examines five regression algorithms, each with its unique capacity to interpret the historical data patterns of natural gas prices:

1. **Linear Regression**: Establishes a baseline for comparison by assuming a straightforward linear relationship between the predictors and the target variable.
   
2. **Polynomial Regression**: Introduces non-linear flexibility to capture complex relationships by extending the linear model with polynomial terms.
   
3. **Decision Trees Regression**: Segregates the predictor space into distinct regions, assigning constant predictive values to each, thus capturing local variations.
   
4. **Random Forest Regression**: Builds on the Decision Tree model, leveraging an ensemble to enhance predictions and mitigate overfitting.
   
5. **Artificial Neural Network Regression**: Implements a deep learning approach to decipher intricate non-linear dependencies.

We utilize a dataset featuring historical natural gas prices and other relevant variables, partitioned into training and testing subsets to objectively assess each model's predictive prowess.

## Results and Insights

The project's findings reveal that model performance is highly data-sensitive, with Polynomial and Decision Tree Regressions demonstrating notable precision in capturing price movements. The Artificial Neural Network, despite facing implementation limitations for visual comparative analysis, displayed potent learning dynamics as evidenced by the loss reduction over epochs. Notably, Decision Tree Regression emerged as notably cost-effective considering its balance between accuracy and computational demand.

## Conclusions

As students in the Master of Science in Data Science program, this project has been an integral component of our academic development, bridging theoretical knowledge with practical application. It has reinforced the criticality of selecting appropriate models tailored to data characteristics and has enriched our understanding of algorithmic efficacy in predictive scenarios.

## Explore

We invite you to delve into the code and findings within this repository, which elucidate the intricacies of forecasting natural gas prices using cutting-edge regression techniques. Your insights and feedback are highly valued as we continue to refine our models and approaches.
