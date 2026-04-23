Missing dates: 0
=== EVENT WINDOW T-TEST ===
Mean BEFORE: 0.3916453265044814
Mean AFTER: 0.4312267657992565
Difference: 0.03958143929477509
T-statistic: -4.781134281049755
P-value: 1.7614534158047206e-06
Optimization terminated successfully.
         Current function value: 0.664879
         Iterations 4
                           Logit Regression Results                           
==============================================================================
Dep. Variable:            any_harmful   No. Observations:                97696
Model:                          Logit   Df Residuals:                    97690
Method:                           MLE   Df Model:                            5
Date:                Thu, 23 Apr 2026   Pseudo R-squ.:                 0.01429
Time:                        16:53:08   Log-Likelihood:                -64956.
converged:                       True   LL-Null:                       -65898.
Covariance Type:            nonrobust   LLR p-value:                     0.000
=================================================================================
                    coef    std err          z      P>|z|      [0.025      0.975]
---------------------------------------------------------------------------------
const            -0.3591      0.020    -17.638      0.000      -0.399      -0.319
post_election     0.8875      0.022     41.219      0.000       0.845       0.930
log_likes        -0.1953      0.014    -14.450      0.000      -0.222      -0.169
log_retweets      0.1642      0.021      7.996      0.000       0.124       0.204
log_replies       0.0761      0.019      4.054      0.000       0.039       0.113
log_quotes        0.0051      0.033      0.154      0.878      -0.059       0.069
=================================================================================

Odds Ratios:
 const            0.698271
post_election    2.429084
log_likes        0.822577
log_retweets     1.178466
log_replies      1.079090
log_quotes       1.005073
dtype: float64
