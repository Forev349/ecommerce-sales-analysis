import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# =====================
# 1. LOAD DATA
# =====================
orders = pd.read_csv(DATA_DIR / 'orders.csv', encoding='utf-8-sig')
customers = pd.read_csv(DATA_DIR / 'customers.csv', encoding='utf-8-sig')
products = pd.read_csv(DATA_DIR / 'products.csv', encoding='utf-8-sig')

# =====================
# 2. DATA QUALITY
# =====================
data = orders.merge(customers, on='customer_id', how='inner')
data = data.merge(products, on='product_id', how='inner')
data['order_date'] = pd.to_datetime(data['order_date'])
data['signup_date'] = pd.to_datetime(data['signup_date'])
data['discount_pct'] = data['discount_pct'].fillna(0)
data['signup_date_error'] = data['signup_date'] > data['order_date']

data_quality = pd.DataFrame({
    'check': [
        'orders rows', 'merged rows', 'full duplicates', 'unique order_id',
        'missing discount', 'signup_date after order_date',
        'future order dates', 'quantity <= 0', 'price <= 0',
        'discount < 0', 'discount > 100'
    ],
    'value': [
        len(orders), len(data), int(data.duplicated().sum()),
        int(data['order_id'].nunique()), int(orders['discount_pct'].isna().sum()),
        int(data['signup_date_error'].sum()), int((data['order_date'] > '2026-08-31').sum()),
        int((data['quantity'] <= 0).sum()), int((data['price'] <= 0).sum()),
        int((data['discount_pct'].fillna(0) < 0).sum()), int((data['discount_pct'].fillna(0) > 100).sum())
    ]
})

# Exclude invalid future dates only from analytical time series; keep raw data intact.
analysis = data[data['order_date'] <= '2026-08-31'].copy()
analysis['gross_value'] = analysis['price'] * analysis['quantity']
analysis['revenue_discounted'] = analysis['gross_value'] * (1 - analysis['discount_pct'] / 100)
analysis['month'] = analysis['order_date'].dt.to_period('M')

# =====================
# 3. REVENUE DYNAMICS
# =====================
potential_monthly = analysis.groupby('month').agg(
    gross_value=('gross_value', 'sum'),
    discounted_value=('revenue_discounted', 'sum')
).reset_index()

actual = analysis[analysis['status'] == 'completed'].copy()
actual_monthly = actual.groupby('month').agg(
    gross_value=('gross_value', 'sum'),
    discounted_value=('revenue_discounted', 'sum')
).reset_index()

# Monthly growth of actual discounted revenue.
actual_monthly['growth_pct'] = actual_monthly['discounted_value'].pct_change() * 100

# =====================
# 4. CUSTOMER ANALYSIS
# =====================
customer_top = actual.groupby('customer_id').agg(
    quantity=('quantity', 'sum'),
    orders=('order_id', 'count'),
    revenue_discounted=('revenue_discounted', 'sum')
).reset_index()

q75 = customer_top['revenue_discounted'].quantile(0.75)
customer_top['large_customer'] = customer_top['revenue_discounted'] > q75
large_customers = customer_top[customer_top['large_customer']].copy()

customer_total_revenue = customer_top['revenue_discounted'].sum()
large_customer_revenue = large_customers['revenue_discounted'].sum()
large_customer_share = large_customer_revenue / customer_total_revenue * 100

# Customer concentration by top 10 as an additional diagnostic.
top10_share = customer_top.nlargest(10, 'revenue_discounted')['revenue_discounted'].sum() / customer_total_revenue * 100

# Monthly revenue from large customers.
large_ids = set(large_customers['customer_id'])
large_monthly = actual[actual['customer_id'].isin(large_ids)].groupby('month').agg(
    large_customer_revenue=('revenue_discounted', 'sum')
).reset_index()

# Compare first 3 months and last 3 months for large-customer activity.
first_months = sorted(actual['month'].unique())[:3]
last_months = sorted(actual['month'].unique())[-3:]
first_large = large_monthly[large_monthly['month'].isin(first_months)]['large_customer_revenue'].sum()
last_large = large_monthly[large_monthly['month'].isin(last_months)]['large_customer_revenue'].sum()
large_period_change_pct = (last_large / first_large - 1) * 100 if first_large else np.nan

# =====================
# 5. PRODUCT ANALYSIS
# =====================
product_analysis = actual.groupby(['product_id', 'product', 'category']).agg(
    quantity=('quantity', 'sum'),
    orders=('order_id', 'count'),
    revenue_discounted=('revenue_discounted', 'sum')
).reset_index()
product_analysis['revenue_share_pct'] = product_analysis['revenue_discounted'] / customer_total_revenue * 100
product_analysis = product_analysis.sort_values('revenue_discounted', ascending=False)

category_analysis = actual.groupby('category').agg(
    quantity=('quantity', 'sum'),
    orders=('order_id', 'count'),
    revenue_discounted=('revenue_discounted', 'sum')
).reset_index()
category_analysis['revenue_share_pct'] = category_analysis['revenue_discounted'] / customer_total_revenue * 100
category_analysis = category_analysis.sort_values('revenue_discounted', ascending=False)

# =====================
# 6. CHANNEL ANALYSIS
# =====================
channel_analysis = actual.groupby('channel').agg(
    orders=('order_id', 'count'),
    customers=('customer_id', 'nunique'),
    quantity=('quantity', 'sum'),
    revenue_discounted=('revenue_discounted', 'sum')
).reset_index()
channel_analysis['avg_order_value'] = channel_analysis['revenue_discounted'] / channel_analysis['orders']
channel_analysis['revenue_share_pct'] = channel_analysis['revenue_discounted'] / customer_total_revenue * 100
channel_analysis = channel_analysis.sort_values('revenue_discounted', ascending=False)

# =====================
# 7. CUSTOMER METRICS
# =====================
customer_metrics = pd.DataFrame({
    'metric': [
        'unique buyers', 'repeat buyers', 'repeat buyer share',
        'average order value', 'median order value', 'orders per buyer'
    ],
    'value': [
        actual['customer_id'].nunique(),
        int((customer_top['orders'] > 1).sum()),
        (customer_top['orders'] > 1).mean() * 100,
        actual['revenue_discounted'].mean(),
        actual['revenue_discounted'].median(),
        len(actual) / actual['customer_id'].nunique()
    ]
})

# =====================
# 8. SIMPLE STATISTICAL / A-B CASE
# =====================
# Compare Ads vs Organic average discounted order value.
# This is an observational comparison, not automatically causal.
ads = actual.loc[actual['channel'] == 'ads', 'revenue_discounted'].dropna()
organic = actual.loc[actual['channel'] == 'organic', 'revenue_discounted'].dropna()

from scipy.stats import ttest_ind
ab_stat, ab_p = ttest_ind(ads, organic, equal_var=False)
mean_ads = ads.mean()
mean_organic = organic.mean()

ab_result = pd.DataFrame({
    'metric': ['ads AOV', 'organic AOV', 'difference ads-organic', 'p_value', 'significant_at_0.05'],
    'value': [mean_ads, mean_organic, mean_ads - mean_organic, ab_p, bool(ab_p < 0.05)]
})

# =====================
# 9. PLOTS
# =====================
plt.figure(figsize=(12, 5))
plt.plot(potential_monthly['month'].astype(str), potential_monthly['gross_value'], label='До скидки')
plt.plot(potential_monthly['month'].astype(str), potential_monthly['discounted_value'], label='После скидки')
plt.xticks(rotation=60)
plt.title('Потенциальный объём заказов по месяцам')
plt.ylabel('Рубли')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'revenue_potential.png', dpi=150)
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(actual_monthly['month'].astype(str), actual_monthly['gross_value'], label='До скидки')
plt.plot(actual_monthly['month'].astype(str), actual_monthly['discounted_value'], label='После скидки')
plt.xticks(rotation=60)
plt.title('Фактическая выручка по месяцам (completed)')
plt.ylabel('Рубли')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'revenue_actual.png', dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(category_analysis['category'], category_analysis['revenue_discounted'])
plt.xticks(rotation=30, ha='right')
plt.title('Выручка по категориям')
plt.ylabel('Рубли')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'category_revenue.png', dpi=150)
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(channel_analysis['channel'], channel_analysis['revenue_discounted'])
plt.title('Выручка по каналам')
plt.ylabel('Рубли')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'channel_revenue.png', dpi=150)
plt.close()

# =====================
# 10. REPORT SUMMARY
# =====================
summary = pd.DataFrame({
    'metric': [
        'actual revenue', 'large customer threshold Q75',
        'large customers count', 'large customers revenue',
        'large customers revenue share %', 'top 10 customer share %',
        'large customers first 3 months revenue',
        'large customers last 3 months revenue',
        'large customers period change %',
        'best category by revenue', 'best channel by revenue',
        'ads vs organic AOV p-value'
    ],
    'value': [
        customer_total_revenue, q75, len(large_customers), large_customer_revenue,
        large_customer_share, top10_share, first_large, last_large,
        large_period_change_pct,
        category_analysis.iloc[0]['category'], channel_analysis.iloc[0]['channel'], ab_p
    ]
})

with pd.ExcelWriter(RESULTS_DIR / 'project_results.xlsx', engine='openpyxl') as writer:
    data_quality.to_excel(writer, sheet_name='data_quality', index=False)
    potential_monthly.to_excel(writer, sheet_name='revenue_potential', index=False)
    actual_monthly.to_excel(writer, sheet_name='revenue_actual', index=False)
    customer_top.to_excel(writer, sheet_name='customers', index=False)
    large_monthly.to_excel(writer, sheet_name='large_customers_time', index=False)
    product_analysis.to_excel(writer, sheet_name='products', index=False)
    category_analysis.to_excel(writer, sheet_name='categories', index=False)
    channel_analysis.to_excel(writer, sheet_name='channels', index=False)
    customer_metrics.to_excel(writer, sheet_name='customer_metrics', index=False)
    ab_result.to_excel(writer, sheet_name='statistics', index=False)
    summary.to_excel(writer, sheet_name='summary', index=False)

print('\n=== SUMMARY ===')
print(summary.to_string(index=False))
print('\n=== DATA QUALITY ===')
print(data_quality.to_string(index=False))
print('\n=== TOP CATEGORIES ===')
print(category_analysis.to_string(index=False))
print('\n=== CHANNELS ===')
print(channel_analysis.to_string(index=False))
print('\n=== TOP CUSTOMERS ===')
print(customer_top.head(10).to_string(index=False))
print('\n=== A/B / STATISTICAL CHECK ===')
print(ab_result.to_string(index=False))
