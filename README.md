# Instacart Frequent Itemset Mining

An industry-survey and reproducible Google Colab project for **Honours – Advanced Data Analytics (216H03C701), IA2, Semester VII, AY 2026–27**. It studies how co-purchase patterns can support complementary-product recommendations at Instacart and maps the problem to **Apriori, FP-Growth, association rules, and recommendation systems**.

> **Important research boundary:** Instacart publicly says its product-pairing service combines co-purchase scoring with machine-learning relevance ranking. It does **not** publicly state that its production system uses Apriori or FP-Growth. This project uses those syllabus algorithms as transparent academic models on Instacart's public 2017 dataset; it does not reverse-engineer or claim to reproduce Instacart's proprietary stack.

## What the notebook does

1. Downloads the public **Instacart Market Basket Analysis** data from Kaggle (or accepts manually uploaded CSV files).
2. Validates and joins orders, products, aisles, and order-product records.
3. Explores basket size, product/aisle frequency, reorder behaviour, and temporal patterns.
4. Builds a bounded, sparse transaction matrix so it can run in Colab without exhausting RAM.
5. Mines frequent itemsets with both **Apriori** and **FP-Growth** and benchmarks their runtime.
6. Generates association rules using support, confidence, lift, leverage, and conviction.
7. Produces “frequently bought together” recommendations for an example basket.
8. Evaluates rule recommendations with a leakage-resistant, per-user last-prior-order holdout using Hit Rate@K, Recall@K, Precision@K and coverage.
9. Discusses industrial scalability, algorithm trade-offs, privacy, bias, modern AI integration, and future improvements.

## Repository structure

```text
.
├── Instacart_Frequent_Itemset_Mining.ipynb  # Colab-ready end-to-end analysis
├── REPORT.md                                # Report-ready industry survey
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Run in Google Colab

Open `Instacart_Frequent_Itemset_Mining.ipynb` in Colab and choose **Runtime → Run all**. The default configuration is intentionally bounded:

```python
MAX_ORDERS = 30_000
TOP_N_PRODUCTS = 250
MIN_SUPPORT = 0.003
MAX_ITEMSET_LEN = 3
```

For Kaggle download, the notebook first tries `kagglehub.competition_download(...)`. If Kaggle authentication or competition acceptance is required, follow the notebook's manual-upload cell and upload the six competition CSV files. Dataset files are deliberately excluded from Git because their distribution is governed by Kaggle's competition rules.

For a quick local run:

```bash
python -m pip install -r requirements.txt
jupyter notebook Instacart_Frequent_Itemset_Mining.ipynb
```

## Algorithm mapping

| Course concept | Project role | Strength | Limitation |
|---|---|---|---|
| Apriori | Baseline frequent-itemset discovery | Simple and interpretable; downward-closure prunes candidates | Candidate explosion and repeated scans at low support |
| FP-Growth | Scalable frequent-itemset discovery | Avoids explicit candidate generation; usually faster on dense/repetitive data | FP-tree construction is less intuitive and can still grow large |
| Association rules | Converts itemsets into product pairings | Explainable support/confidence/lift scores | Correlation, not causation; popularity and promotion bias |
| Recommendation systems | Ranks candidate complements | Directly maps patterns to user-facing suggestions | Pure co-occurrence is not personalized and has cold-start issues |

For itemset \(X\) and consequent \(Y\):

- **Support:** \(P(X \cup Y)\)
- **Confidence:** \(P(Y\mid X)=P(X\cup Y)/P(X)\)
- **Lift:** \(P(Y\mid X)/P(Y)\); values above 1 indicate positive association

High lift alone is unsafe on rare items, so the notebook enforces minimum support and ranks rules with multiple signals.

## Industrial evidence and findings

- Instacart's current Storefront documentation says complementary-item pairing combines a **machine-learning relevance score** and a **co-purchasing score**, and uses purchase patterns from fulfilled orders. This is the strongest direct evidence connecting the business application to market-basket analytics.[^1]
- The public dataset contains more than three million anonymized grocery orders from over 200,000 users and was released for the 2017 Kaggle competition.[^2]
- Instacart reports that its ML applications span recommendations, search, personalization, ads, fraud and logistics, with a shared ML platform built to train, deploy and serve models at scale.[^3]
- Its distributed-ML architecture is designed for CPU/GPU scalability, utilization, throughput and reliability across many production workloads.[^4]
- In February 2026, Instacart described an AI-native discovery platform where LLM-generated placements are filtered/evaluated with traditional models (including a fine-tuned DeBERTa relevance classifier), while conventional ranking models remain candidates for reward-model integration.[^5]

Together, the evidence suggests a practical layered architecture: co-purchase statistics are useful for candidate generation and explainability; learned ranking adds context, personalization and business constraints; generative/foundation models can improve discovery content and semantic understanding. That architecture is an inference from the cited public material, not a claim about undisclosed implementation details.

## Critical analysis

Apriori is ideal for explaining the syllabus concept but is not a plausible sole solution at Instacart scale: the product catalog and number of possible combinations make candidate enumeration expensive. FP-Growth is a stronger batch-mining baseline. In production, alternatives such as distributed FP-Growth, approximate co-occurrence counts, item embeddings, sequential models, gradient-boosted ranking, or a two-stage retrieval-and-ranking system could perform better.

The experiment's rules remain global and observational. They do not directly model sequence, store availability, geography, price, promotions, dietary preferences, seasonality, or causality. Offline metrics also do not prove incremental basket growth; an online randomized experiment would be required. Responsible deployment should include privacy protection, frequency thresholds, bias audits, diversity constraints, out-of-stock filtering, and monitoring for drift.

## Reproducibility notes

- Fixed random seed and explicit configuration are at the top of the notebook.
- No raw customer identifiers or CSV data are committed.
- The notebook reports library versions, memory use, sample sizes, thresholds, timings, and metrics.
- Outputs depend on the selected sample and thresholds; the README intentionally does not invent result values before execution.

## AI usage declaration

ChatGPT/Codex assisted with research planning, source discovery, code structure, explanation, and documentation. Claims were checked against Instacart documentation/engineering posts, Kaggle's dataset page, and the original dataset announcement. The main correction made during verification was to avoid claiming that Instacart uses Apriori or FP-Growth in production: public evidence supports co-purchase scoring and ML ranking, but not those exact proprietary implementation details. The authors should run the notebook, inspect every output, verify the cited pages, and edit the report in their own words before submission.

## References

[^1]: Instacart Docs, “Recommendations,” complementary items and scoring. https://docs.instacart.com/storefront/concepts/recommendations/
[^2]: Instacart, “3 Million Instacart Orders, Open Sourced” (2017), archived dataset description: https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2 and Kaggle competition: https://www.kaggle.com/competitions/instacart-market-basket-analysis/data
[^3]: Instacart, “Supercharging ML/AI Foundations at Instacart” (2023). https://company.instacart.com/tech-innovation/supercharging-ml-ai-foundations-at-instacart
[^4]: Instacart, “Distributed Machine Learning at Instacart” (2023). https://company.instacart.com/tech-innovation/distributed-machine-learning-at-instacart
[^5]: Instacart, “Our Early Journey to Transform Instacart's Discovery Recommendations with LLMs” (2026). https://company.instacart.com/tech-innovation/our-early-journey-to-transform-instacart-s-discovery-recommendations-with-llms
[^6]: Agrawal, Imieliński & Swami, “Mining Association Rules between Sets of Items in Large Databases” (SIGMOD 1993). https://doi.org/10.1145/170035.170072
[^7]: Han, Pei & Yin, “Mining Frequent Patterns without Candidate Generation” (SIGMOD 2000). https://doi.org/10.1145/342009.335372

## Academic integrity

Use this repository as a reproducible foundation, not as a substitute for group understanding. Add team details, execute the notebook, interpret the actual outputs, and comply with your university's and Kaggle's rules.

