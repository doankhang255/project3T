# Internship Roadmap for Bui Doan Khang

## 1. What the internship requirements actually ask for

From the internship document, your 5 targets are:

1. Preprocess and perform EDA on Vietnam stock market news/event data.
2. Build an NLP sentiment model for stock market news.
3. Compute an Equity Sentiment Index and analyze its impact on VN-Index and stocks.
4. Backtest trading signals based on the sentiment index.
5. Write the final report and compare with the macro sentiment work of your colleague.

Right now, you only need to finish target 1 and prepare the foundation for target 2.

## 2. Assessment of your current EDA notebook

Your current `EDA.ipynb` already has some useful parts:

- API extraction for `dataset_stock_events`
- Basic dtype conversion
- Date anomaly checking
- Duplicate removal
- Some event title normalization ideas
- First attempt at sentiment rules

But there is one core mismatch with the internship objective:

- The old `event_group_map` groups events into business categories such as
  `Financial Reporting`, `Corporate Action`, `Governance`, `Listing Status`.
- That grouping is useful for descriptive EDA.
- It is not the correct target design for an equity sentiment project.

For equity sentiment, the label design should be:

- `positive`
- `negative`
- `neutral`

And optionally during review:

- `mixed`
- `unknown`

## 3. What should requirement 1 contain

Requirement 1 should not stop at "download and clean data".
It should end with a clearly explained, sentiment-ready dataset.

Suggested structure:

1. Data source overview
2. Column meaning
3. Data type normalization
4. Missing-value analysis
5. Date consistency checks
6. Duplicate analysis
7. Event code normalization
8. Event title text inspection
9. First-pass sentiment labeling with rules
10. Coverage review of `positive/negative/neutral`
11. Export-ready clean dataset for later modeling

The file `EDA_core.py` is built exactly for this stage.

## 4. What should requirement 2 contain

Requirement 2 is not just "train a model".
You first need a defensible labeled dataset.

Recommended flow:

1. Build weak labels from `event_title` and selected `event_code` priors.
2. Review `mixed` and `unknown` rows manually.
3. Create a final training set with only `positive/negative/neutral`.
4. Split train/validation/test carefully.
5. Start with baseline models:
   - TF-IDF + Logistic Regression
   - TF-IDF + Linear SVM
6. If needed, move to PhoBERT or multilingual FinBERT-style transformer models.
7. Evaluate with:
   - accuracy
   - macro F1
   - confusion matrix

Important:
- Requirement 2 says accuracy should be at least 80%.
- Do not rely only on accuracy if labels are imbalanced.
- Macro F1 should be shown as supporting evidence.

## 5. Suggested direction for all 5 requirements

### Requirement 1
- Finish the clean EDA pipeline.
- Produce a sentiment-ready dataset.
- Show why sentiment labels are more suitable than event groups.

### Requirement 2
- Create a labeled training dataset from requirement 1.
- Train baseline NLP models.
- Compare model performance and error types.

### Requirement 3
- Aggregate sentiment by date or week.
- Build an Equity Sentiment Index, for example:
  `sentiment_index = (positive_count - negative_count) / total_count`
- Compare the index with VN-Index return, volume, and volatility.

### Requirement 4
- Convert sentiment index into simple signals such as:
  buy when index > threshold,
  reduce exposure when index < threshold.
- Backtest on VN-Index or selected stocks.
- Report return, Sharpe, drawdown, hit rate.

### Requirement 5
- Write the full story:
  data -> labeling -> model -> index -> market impact -> backtest.
- Compare equity sentiment against macro sentiment work.

## 6. Notes about the new file `EDA_core.py`

This new file is intentionally separated from `EDA.ipynb`.

Why:

- avoid conflict with your current notebook
- keep old work for reference
- make the core logic easier to read and reuse later
- avoid accidental API calls or data writes

The file does:

- validate expected columns
- standardize dtypes
- clean implausible dates
- remove duplicates
- normalize `event_code`
- create `sentiment_label`
- produce EDA summary tables
- create a training-ready subset

The file does not:

- call your private API key
- download data
- modify source data
- save files automatically

## 7. Practical next step for you

Your immediate next step should be:

1. Load your real dataset into a DataFrame in notebook.
2. Import `run_eda_core` from `EDA_core.py`.
3. Run it on a copied dataset.
4. Inspect:
   - `quality_report["missing_summary"]`
   - `sentiment_report["label_distribution"]`
   - rows with `sentiment_label` equal to `mixed` or missing
5. Expand the regex rules based on your real Vietnamese/English event titles.

## 8. One important caveat

Your current preview dataset is `stock_events`, not pure news articles.

That means:

- some rows are genuinely neutral disclosures
- sentiment may be inferred from event semantics, not only text tone
- you should present this honestly in the report:
  this is "equity-related event sentiment" or "market-relevant disclosure sentiment"
  unless your later dataset includes richer news text

That framing will make your methodology more defensible.
