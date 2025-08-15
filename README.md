# Lasso Regression
- Lasso minimizes the residual sum of squares (RSS) plus a penalty proportional to the absolute values of the coefficients:
![WhatsApp Image 2025-08-13 at 09 22 39_117b8750](https://github.com/user-attachments/assets/fd949611-c1e7-4942-aeed-5f3501ed8edc)
- where λ (or α) controls the strength of the penalty https://www.statisticshowto.com/lasso-regression
- L₁ regularization is key: unlike L₂ (ridge), Lasso can force coefficients exactly to zero, enabling built-in feature selection https://www.statisticshowto.com/lasso-regression
<img width="988" height="505" alt="image" src="https://github.com/user-attachments/assets/3050c0ad-cc1c-4c85-be30-896577ac9eab" />
<img width="822" height="505" alt="image" src="https://github.com/user-attachments/assets/5867cfc2-7e63-44f3-8389-03385e05f32a" />

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
<img width="1152" height="759" alt="image" src="https://github.com/user-attachments/assets/00d49f27-19d0-4b4c-87a5-4ad36841499e" />


##### 2) Higher coefficients are affected more
- L₂ (ridge) shrinks proportionally to size -> bigger coefficients shrink more (multiplicative shrinkage).
- L₁ (Lasso) applies a fixed absolute shrinkage (soft-threshold): every nonzero coefficient is reduced by roughly the same amount (until it hits zero).
-> Small coefficients are more likely to be killed; large ones shrink but usually stay nonzero.

- So, while the penalty value ∥w∥1 grows with coefficient size (discouraging large magnitudes), the shrinkage rule itself is not “more shrink for bigger coefs” the way ridge is—it’s a constant knock-down per coordinate.
<img width="1230" height="659" alt="image" src="https://github.com/user-attachments/assets/c11189a1-5e14-4f30-ae0a-d2524bbd8252" />


##### 3) Impact on bias and variance
- As α increases:
  - Bias increase (we force coefficients toward/at zero, underfitting risk if too high).
  - Variance decrease (simpler, sparser models that generalize better, especially in high dimensional settings).

- The test error vs α is typically U-shaped. Use cross-validation to pick α that balances bias–variance.
- With correlated predictors, Lasso tends to pick one and drop the rest (can make selection unstable across resamples). If thats a problem, consider Elastic Net (L1+L2) to share weight across correlated features.
<img width="547" height="432" alt="image" src="https://github.com/user-attachments/assets/e0b80e70-2383-42a5-b887-77d5888f59db" />


##### 4) Effect of regularization on the loss function

<img width="1001" height="1002" alt="image" src="https://github.com/user-attachments/assets/137239d6-f1a3-44f0-9a86-4ac51b10db89" />


# Why does Lasso Regression show sparsity?

##### Sparsity means that when you increase the value of λ (lambda) in Lasso Regression, the coefficients for some input features (columns) become exactly zero.This means Lasso is effectively removing those features from the model (feature selection).
- If λ is large → many coefficients shrink exactly to 0
- If coefficient w -> 0, then that feature is not used in prediction.

##### You wrote that in Ridge Regression, even if you increase alpha (λ) a lot, the coefficients get smaller but do not become exactly zero.
- Ridge only shrinks values toward zero, but never exactly zero.
- Example: Even at large α, you might see coefficients like 0.05, 0.01, etc., but not 0.

##### In Lasso Regression:
- Even with a small increase in α, you may start seeing some coefficients become exactly zero.
- As α increases further, more and more coefficients will become zero.
- Eventually, only the most important features remain with non-zero coefficients.  

###### The main reason is the L1 penalty used by Lasso:

![WhatsApp Image 2025-08-15 at 19 04 53_a497b489](https://github.com/user-attachments/assets/cbf1376b-0f6b-445e-957a-0a3bd28daa5b)

- The absolute value term ∣wi∣ creates a sharp corner at 0 in the cost function.
- This "corner" makes it easier for optimization to push coefficients exactly to zero during minimization.
- This is why Lasso is capable of automatic feature selection.

###### In contrast, Ridge Regression uses L2 penalty:

![WhatsApp Image 2025-08-15 at 19 04 53_9561aa5b](https://github.com/user-attachments/assets/3cc02c92-fae0-48b0-9f2a-2c76cc845c58)

- The squared term is smooth and round at 0, so it shrinks coefficients but does not make them exactly zero.
