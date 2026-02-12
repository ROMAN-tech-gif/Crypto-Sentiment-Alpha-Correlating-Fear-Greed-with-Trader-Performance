# Crypto-Sentiment-Alpha-Correlating-Fear-Greed-with-Trader-Performance
This repository provides a data-driven investigation into the relationship between market sentiment and trading behavior. By cross-referencing seven years of the Crypto Fear &amp; Greed Index with 211,000+ transaction logs, the project identifies how emotional extremes influence profitability, position sizing, and risk management.
# Crypto Sentiment vs. Trading Behavior

This project analyzes whether the **Fear & Greed Index** drives specific trading behaviors and if "Pro" traders handle sentiment shifts differently than "Retail" traders.
## Analysis Steps
1. **Data Alignment**: Merged 7 years of sentiment data with 211k+ trade logs.
2. **Feature Engineering**: Created daily PnL, win rates, and size metrics.
3. **Segmentation**: Categorized users into behavioral archetypes (Pro vs Retail).
4. **Modeling**: Built a classifier to predict profitability based on market fear levels.

## Requirements
- `pandas`, `seaborn`, `scikit-learn`
