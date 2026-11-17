import math

# Unnormalized probabilities from test_tool_logits:
# Math query:
p_py_math = 0.9982
p_web_math = 0.0001

# Web query:
p_py_web = 0.8181
p_web_web = 0.0170

# Bayesian prior correction: In training data, Python:Web was 625:50 = 12.5
# log-odds adjustment = ln(12.5) = 2.5257
prior_ratio = 625.0 / 50.0

# Adjusted odds: Odds_adj = (P(web)/P(py)) * prior_ratio
odds_math = (p_web_math / p_py_math) * prior_ratio
p_web_math_adj = odds_math / (1.0 + odds_math)

odds_web = (p_web_web / p_py_web) * prior_ratio
p_web_web_adj = odds_web / (1.0 + odds_web)

print(f"Math query: adjusted P(web)={p_web_math_adj*100:.2f}%, adjusted P(python)={(1-p_web_math_adj)*100:.2f}%")
print(f"Web query:  adjusted P(web)={p_web_web_adj*100:.2f}%, adjusted P(python)={(1-p_web_web_adj)*100:.2f}%")
