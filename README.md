# Lasso Regression
- Lasso minimizes the residual sum of squares (RSS) plus a penalty proportional to the absolute values of the coefficients:
![WhatsApp Image 2025-08-13 at 09 22 39_117b8750](https://github.com/user-attachments/assets/fd949611-c1e7-4942-aeed-5f3501ed8edc)
- where λ (or α) controls the strength of the penalty https://www.statisticshowto.com/lasso-regression
- L₁ regularization is key: unlike L₂ (ridge), Lasso can force coefficients exactly to zero, enabling built-in feature selection https://www.statisticshowto.com/lasso-regression
---
##### Feature Selection & Sparsity
- As λ increases, Lasso shrinks more coefficients to zero, resulting in a sparse model that includes only the most predictive features https://www.statisticshowto.com/lasso-regression
- It handles multicollinearity by typically selecting one variable from a group of correlated ones and discarding the rest.
---
###### 5 key points about lasso regression. 
1) How are coefficient affected
2) Higher coefficent are affected more.
3) Impact on bias and variance.
4) Effect of regularization of loss function. explain this key points step by step

##### 1) How are coefficient affected
- Lasso adds an L₁ penalty to linear regression:
minimize 1/2n∥y−Xw∥2+𝛼∥w∥1
- This penalty pulls each coefficient toward zero; many become exactly zero (automatic feature selection).
- In an orthonormal/simple setting, each coordinate update is soft-thresholding:
wj<-sign(zj) max(∣zj∣−𝛼,0)
(where zj is the least-squares/partial-residual estimate for feature j).
If ∣zj∣≤α, the coefficient is set to 0.

##### 2) Higher coefficients are affected more
- L₂ (ridge) shrinks proportionally to size -> bigger coefficients shrink more (multiplicative shrinkage).
- L₁ (Lasso) applies a fixed absolute shrinkage (soft-threshold): every nonzero coefficient is reduced by roughly the same amount (until it hits zero).
-> Small coefficients are more likely to be killed; large ones shrink but usually stay nonzero.

- So, while the penalty value ∥w∥1 grows with coefficient size (discouraging large magnitudes), the shrinkage rule itself is not “more shrink for bigger coefs” the way ridge is—it’s a constant knock-down per coordinate.

##### 3) Impact on bias and variance
- As α increases:
  - Bias increase (we force coefficients toward/at zero, underfitting risk if too high).
  - Variance decrease (simpler, sparser models that generalize better, especially in high dimensional settings).

- The test error vs α is typically U-shaped. Use cross-validation to pick α that balances bias–variance.
- With correlated predictors, Lasso tends to pick one and drop the rest (can make selection unstable across resamples). If thats a problem, consider Elastic Net (L1+L2) to share weight across correlated features.

##### 4) Effect of regularization on the loss function




