## Quantium Virtual Internship

### Project Overview
This project create a measure to compare  different control stores to each of the trial stores. The analysis helps determine whether the trial intervention had a significant impact on sales performance and identifies the key drivers of any observed changes.

### Problem Statement
Quantium's client just did a trial (like promotion) to their three stores (store number 77, store number 86, and store number 88), They want to know whether their each trial stores successfully increase the sales?. But there is a problem, they're realized that the fluctuating sales happens beacause many things (holiday, weekend, weather, traffic, trend, etc), so they dont know for the example, store number 77 increase sales beacuse from promotion or other things. While they want to analyze their promotion impact or not for the sales.

### Objective
Assess the effectiveness of new store baseline implemented to trial store

### Approach
We take all that client stores then compare to each trial stores (also called control store) during before the trial period, then the output is the most best control store from each trial stores. And last, we compare that best control store to each trial store during trial period that we determine before.

### Key Question
Did the trial intervention significantly impact sales, and what drove the change?

### Dataset
Source: QVI_data.csv (Quantium Virtual Internship)
Key Fields:

LYLTY_CARD_NBR - Customer loyalty card number
DATE - Transaction date
STORE_NBR - Store identifier
TXN_ID - Transaction ID
PROD_NBR - Product number
PROD_NAME - Product name
PROD_QTY - Quantity purchased
TOT_SALES - Total sales amount
PACK_SIZE - Product pack size
BRAND - Product brand
LIFESTAGE - Customer life stage
PREMIUM_CUSTOMER - Customer segment

### Analysis Framework
1. Trial Configuration

-Trial Stores: 77, 86, 88
-Trial Period: February 2019 - April 2019
-Pre-Trial Period: 12 months prior (for control selection)

2. Key Metrics

-Total Sales Revenue - Monthly total sales per store
-Number of Customers - Unique customers per month
-Transactions per Customer - Average purchase frequency

### Methodology
1. Data Preparation

Aggregate transaction data to monthly store-level metrics
Calculate three key performance indicators for each store-month
Filter data into pre-trial and trial periods

2. Control Store Selection
Dual-Metric Approach:
A. Pearson Correlation
Measures similarity in trend patterns between trial and control stores.

Range: -1 to +1 (1 = perfect correlation)
Applied to monthly time series of each metric

B. Magnitude Distance
Measures similarity in absolute values.

Formula: 1 - (observed_distance - min_distance) / (max_distance - min_distance)
Normalized to 0-1 scale (1 = most similar)

C. Combined Scoring
Final Score = Σ [(correlation × 0.5 + magnitude × 0.5) × metric_weight]

Metric Weights:
- Total Sales: 40%
- Number of Customers: 40%
- Transactions per Customer: 20%

Selection Criteria:

Highest final score = best control store
Must not be a trial store
Must have complete data for pre-trial period

### Results and Explanation
##### 1. Results (Initial Findings) Trial Stores
- Store 77 records the highest sales (+29%), and its parallel also increasing the number of customers (23,48%)
- Store 88 records all good records for sales (12,3%), customer growth (5,76%), and highest possitive transaction (7%)
- Store 86 records the lowest records for sales (9,7%), and second highest customer growth (13,5%)

#### 2. Recommendation & Strategy
- For store 77:
  Focus for campaign accusission new customer, instance; digital advertisement, voucher for first purchase, or collaboration or partnership to influencer
- For store 86:
  Re-evaluate the promotion, beacause the growth customer did not signifcantlly grow, as well as customer growth and transaction that decrease, perhaps the strategy did'n      interest for old customer and also to new customer. Try to variate the bundling promo, reward point, or specifically personlaization based on purchased history customer.
- For store 88:
  This is the most ideal model promotion from all 2 above because all metrics show possitive growth. The promotion successfully in driving loyal customers to shop           frequently. Try to replicate this strategy promotion to other stores to obtain same gained



