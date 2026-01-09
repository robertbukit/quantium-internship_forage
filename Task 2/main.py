import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

## CREATE MONTHLY STORE METRICS
def monthly_metrics(df):
    """
    Aggregate monthly sales store from 3 metrics:
    1. Total sales revenue
    2. Total Number of Customers
    3. Average Transactions per Customer
    """
    monthly_data = df.groupby(['STORE_NBR', 'YEAR_MONTH']).agg({
        'TOT_SALES': 'sum',
        'LYLTY_CARD_NBR': 'nunique',
        'TXN_ID': 'count'
    }).reset_index()

    # calculate average transcation per customer
    monthly_data['AVG_TXN_PER_CUST'] = (monthly_data['TXN_ID'] / monthly_data['LYLTY_CARD_NBR'])

    # convert YEAR_MONTH back to datetime for consistency
    monthly_data['YEAR_MONTH'] = monthly_data['YEAR_MONTH'].dt.to_timestamp()

    print("\nMonthly metrics created:")
    print(monthly_data.head())

    return monthly_data


## CALCULATE CORRELATION
def calc_corr(trial_store, control_store, measure_data, metric, pre_trial_end):
    """
    Calculate pearson correlation for pre trial period

    Parameters:
    - trial_store: Trial store number
    - control_store: Control store number
    - measure_data: DataFrame dengan monthly metrics
    - metric: 'TOT_SALES', 'LYLTY_CARD_NBR', atau 'TXN_ID'
    - pre_trial_end: End date untuk pre-trial period
    """
    # filter pre-trial period only
    pre_trial = measure_data[measure_data['YEAR_MONTH'] <= pre_trial_end]

    # get trial store data
    trial_data = pre_trial[pre_trial['STORE_NBR'] == trial_store][['YEAR_MONTH', metric]].sort_values('YEAR_MONTH')

    # get control store data
    control_data = pre_trial[pre_trial['STORE_NBR'] == control_store][['YEAR_MONTH', metric]].sort_values('YEAR_MONTH')

    # merge trial store with control store
    merged = trial_data.merge(control_data, on='YEAR_MONTH',
                              suffixes=('_trial', '_control'))
    
    if len(merged) < 3:  # Need at least 3 points for correlation
        return 0
    
    # Calculate Pearson correlation
    corr, _ = pearsonr(merged[f'{metric}_trial'], 
                       merged[f'{metric}_control'])
    
    return corr

## CALCULATE MAGNITUDE DISTANCE
def calc_magnitude_dis(trial_store, control_store, measure_data, metric, pre_trial_end):
    """
    Calculate normalized magnitude distance 
    Formula: 1 - (observed_distance - min_distance) / (max_distance - min_distance)
    """
    # filter pre-trial period only
    pre_trial = measure_data[measure_data['YEAR_MONTH'] <= pre_trial_end]

    # calculate mean for trial dan control store
    trial_mean = pre_trial[pre_trial['STORE_NBR'] == trial_store][metric].mean()
    control_mean = pre_trial[pre_trial['STORE_NBR'] == control_store][metric].mean()

    # calculate absolute distance
    distance = abs(trial_mean - control_mean)

    return distance

def normalize_magnitude_scores(distances_dict):
    """
    Normalize magnitude distances to 0 - 1 scale
    """
    distances = list(distances_dict.values())
    min_dist = min(distances)
    max_dist = max(distances)

    normalized = {}
    for store, dist in distances_dict.items():
        if max_dist - min_dist == 0:
            normalized[store] = 1
        else:
            normalized[store] = 1 - (dist - min_dist) / (max_dist - min_dist)
    
    return normalized

## FIND CONTROL STORES
def find_control_store(trial_store, measure_data, pre_trial_end, trial_stores, metric_weights=None):
    """
    Main function to find best control store

    Parameters:
    - trial_store: trial store number
    - measure_data: DataFrame with monthly metrics
    - pre_trial_end: End date for pre-trial period
    - trial_stores: list of trial store numbers to exclude from control candidates
    - metric_weights: dictionary with weights for each metric
        {'TOT_SALES': O.4, 'LYLTY_CARD_NBR': 0.4, 'TXN_ID': 0.2}
    """

    if metric_weights is None:
        metric_weights = {'TOT_SALES': 0.4, 'LYLTY_CARD_NBR': 0.4, 'TXN_ID': 0.2}

    # Get all control store candidates (exclude trial stores)
    all_stores = measure_data['STORE_NBR'].unique()
    control_candidates =[s for s in all_stores if s not in trial_stores]

    print(f"\nFinding control store for Trial Store {trial_store}")
    print(f"Number of Control candidates: {len(control_candidates)}")

    result = []
    for control_store in control_candidates:
        scores = {}

        for metric, weight in metric_weights.items():
            # Correlation score
            corr = calc_corr(trial_store, control_store, measure_data, metric, pre_trial_end)
            scores[f"{metric}_corr"] = corr
        
        result.append({
            'control_store': control_store, **scores
        })

    # Convert to DataFrame
    result_df = pd.DataFrame(result)

    # Calculate magnitude distances
    for metric in metric_weights.keys():
        distances = {}
        for control_store in control_candidates:
            dist = calc_magnitude_dis(trial_store, control_store, measure_data, metric, pre_trial_end)
            distances[control_store] = dist

        # Normalize distances
        normalized_dist = normalize_magnitude_scores(distances)
        result_df[f'{metric}_mag'] = result_df['control_store'].map(normalized_dist)

    # Calculate combined score
    for metric, weight in metric_weights.items():
        result_df[f"{metric}_score"] = (
            (result_df[f"{metric}_corr"] * 0.5) +
            (result_df[f"{metric}_mag"] * 0.5)
        ) * weight

    # Calculate final score
    score_columns = [f'{metric}_score' for metric in metric_weights.keys()]
    result_df['final_score'] = result_df[score_columns].sum(axis=1)
    
    # Sort by final score
    result_df = result_df.sort_values('final_score', ascending=False)
    
    # Get best control store
    best_control = result_df.iloc[0]['control_store']

    print(f"\nBest Control Store: {int(best_control)} "
          f"(Score: {result_df.iloc[0]['final_score']:.4f})")
    
    return int(best_control), result_df

## VISUALIZE PRE-TRIAL COMPARISON
def plot_pre_trial_comparison(trial_store, control_store, measure_data, 
                              pre_trial_end, metrics=['TOT_SALES', 'LYLTY_CARD_NBR']):
    """
    Plot time series comparison untuk pre-trial period
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))
    
    if len(metrics) == 1:
        axes = [axes]
    
    pre_trial = measure_data[measure_data['YEAR_MONTH'] <= pre_trial_end]
    
    for idx, metric in enumerate(metrics):
        trial_data = pre_trial[pre_trial['STORE_NBR'] == trial_store]
        control_data = pre_trial[pre_trial['STORE_NBR'] == control_store]
        
        axes[idx].plot(trial_data['YEAR_MONTH'], trial_data[metric], 
                      marker='o', label=f'Trial {trial_store}', linewidth=2)
        axes[idx].plot(control_data['YEAR_MONTH'], control_data[metric], 
                      marker='s', label=f'Control {control_store}', linewidth=2)
        
        axes[idx].set_title(f'{metric} - Pre-Trial Period', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Month')
        axes[idx].set_ylabel(metric)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'pre_trial_comparison_store_{trial_store}.png', dpi=300, bbox_inches='tight')
    plt.show()

## ANALYZE TRIAL PERIOD
def analyze_trial_period(trial_store, control_store, measure_data,
                         trial_start, trial_end):
    """
    Analyze trial period performnace that answer the problem business by Zilinka,
    "Is the trial store's performance successful significant compared to the control strore? 
    and why?"
    """
    print(f"Trial Period Analysis: Store {trial_store} (trial) vs Store {control_store} (control)")
    
    # filter trial period data
    trial_period = measure_data[(measure_data['YEAR_MONTH'] >= trial_start) & (measure_data['YEAR_MONTH'] <= trial_end)]

    # get trial store data
    trial_data = trial_period[trial_period['STORE_NBR'] == trial_store]

    # get control store data
    control_data = trial_period[trial_period['STORE_NBR'] == control_store]

    # calculate totals
    results = {
        'metric': [],
        'trial_total': [],
        'control_total': [],
        'difference': [],
        'pctg_difference': []
    }

    metrics = ['TOT_SALES', 'LYLTY_CARD_NBR', 'AVG_TXN_PER_CUST']

    for metric in metrics:
        trial_total = trial_data[metric].sum() if metric != 'AVG_TXN_PER_CUST' else trial_data[metric].mean()
        control_total = control_data[metric].sum() if metric != 'AVG_TXN_PER_CUST' else control_data[metric].mean()
        diff = trial_total - control_total
        pctg_diff = (diff / control_total) * 100 if control_total != 0 else np.nan

        results['metric'].append(metric)
        results['trial_total'].append(trial_total)
        results['control_total'].append(control_total)
        results['difference'].append(diff)
        results['pctg_difference'].append(pctg_diff)

    results_df = pd.DataFrame(results)
    print("\nTrial Period Performance Analysiss:")
    print(results_df.to_string(index=False))

    # identify key drivers
    print("\nKey Insights:")

    sales_pct = results_df[results_df['metric'] == 'TOT_SALES']['pctg_difference'].values[0]
    cust_pct = results_df[results_df['metric'] == 'LYLTY_CARD_NBR']['pctg_difference'].values[0]
    txn_pct = results_df[results_df['metric'] == 'AVG_TXN_PER_CUST']['pctg_difference'].values[0]

    if abs(sales_pct) > 5:
        print(f"• Sales difference: {sales_pct:+.1f}% vs control store")
        
        if abs(cust_pct) > abs(txn_pct):
            print(f"• PRIMARY DRIVER: Customer count ({cust_pct:+.1f}%)")
            print(f"• Secondary factor: Transactions per customer ({txn_pct:+.1f}%)")
        else:
            print(f"• PRIMARY DRIVER: Transactions per customer ({txn_pct:+.1f}%)")
            print(f"• Secondary factor: Customer count ({cust_pct:+.1f}%)")
    else:
        print("• No significant difference in sales performance")
    
    return results_df

## VISUALIZE TRIAL PERIOD ANALYSIS
def plot_trial_period_analysis(trial_store, control_store, measure_data,
                               pre_trial_end, trial_start, trial_end):
    """
    Plot full time series with trial period highlieted
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    metrics = ['TOT_SALES', 'LYLTY_CARD_NBR']

    for idx, metric in enumerate(metrics):
        trial_data = measure_data[measure_data['STORE_NBR'] == trial_store]
        control_data = measure_data[measure_data['STORE_NBR'] == control_store]
        
        # Plot full time series
        axes[idx].plot(trial_data['YEAR_MONTH'], trial_data[metric], 
                      marker='o', label=f'Trial {trial_store}', linewidth=2)
        axes[idx].plot(control_data['YEAR_MONTH'], control_data[metric], 
                      marker='s', label=f'Control {control_store}', linewidth=2)
        
        # Highlight trial period
        axes[idx].axvspan(trial_start, trial_end, alpha=0.2, color='yellow', 
                         label='Trial Period')
        axes[idx].axvline(pre_trial_end, color='red', linestyle='--', 
                         alpha=0.5, label='Trial Start')
        
        axes[idx].set_title(f'{metric} - Full Period', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Month')
        axes[idx].set_ylabel(metric)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'trial_analysis_store_{trial_store}.png', dpi=300, bbox_inches='tight')
    plt.show()




