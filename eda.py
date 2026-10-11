"""
=============================================================================
Credit Card Fraud Detection - Exploratory Data Analysis (EDA)
File: eda.py
Author: Senior ML & Full-Stack Developer / College Project Mentor
Description:
    Loads the historical credit card transaction dataset, calculates detailed
    statistical summaries, and produces analytical visualizations showing class
    distribution, transaction amount distribution, and time distributions.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/script plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for publication/dashboard quality charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

def run_eda(dataset_path="dataset/creditcard.csv", output_dir="reports/plots", static_dir="static/img/plots"):
    """
    Executes complete Exploratory Data Analysis and generates visual plots.
    """
    print("=" * 70)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Verify dataset existence
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {dataset_path}")

    print(f"Loading dataset from: {dataset_path} ...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset successfully loaded. Total rows: {len(df):,}, Total columns: {df.shape[1]}")

    # Ensure output folders exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 2. Key Statistics & Numbers
    total_records = len(df)
    missing_vals = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    class_counts = df['Class'].value_counts()
    legit_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_pct = (fraud_count / total_records) * 100
    legit_pct = (legit_count / total_records) * 100

    print("\n--- DATA QUALITY & IMBALANCE METRICS ---")
    print(f"Total Transactions     : {total_records:,}")
    print(f"Missing Values         : {missing_vals}")
    print(f"Duplicate Rows         : {duplicates:,} ({duplicates / total_records * 100:.3f}%)")
    print(f"Legitimate Transactions: {legit_count:,} ({legit_pct:.4f}%)")
    print(f"Fraudulent Transactions: {fraud_count:,} ({fraud_pct:.4f}%)")
    print(f"Imbalance Ratio        : 1 fraud per {legit_count // fraud_count:,} legitimate transactions")

    # Amount statistics
    amount_legit = df[df['Class'] == 0]['Amount']
    amount_fraud = df[df['Class'] == 1]['Amount']

    print("\n--- TRANSACTION AMOUNT COMPARISON ---")
    print(f"Legitimate Mean Amount: €{amount_legit.mean():.2f} (Median: €{amount_legit.median():.2f}, Max: €{amount_legit.max():.2f})")
    print(f"Fraudulent Mean Amount: €{amount_fraud.mean():.2f} (Median: €{amount_fraud.median():.2f}, Max: €{amount_fraud.max():.2f})")

    # Save summary statistics to JSON for Flask and Dashboard use
    eda_summary = {
        "total_records": total_records,
        "missing_values": missing_vals,
        "duplicate_rows": duplicates,
        "legitimate_count": legit_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_pct, 4),
        "legit_percentage": round(legit_pct, 4),
        "amount_legit": {
            "mean": round(float(amount_legit.mean()), 2),
            "median": round(float(amount_legit.median()), 2),
            "std": round(float(amount_legit.std()), 2),
            "max": round(float(amount_legit.max()), 2),
            "min": round(float(amount_legit.min()), 2)
        },
        "amount_fraud": {
            "mean": round(float(amount_fraud.mean()), 2),
            "median": round(float(amount_fraud.median()), 2),
            "std": round(float(amount_fraud.std()), 2),
            "max": round(float(amount_fraud.max()), 2),
            "min": round(float(amount_fraud.min()), 2)
        }
    }

    with open("reports/eda_summary.json", "w") as f:
        json.dump(eda_summary, f, indent=4)
    print("\n[INFO] Summary metrics saved to reports/eda_summary.json")

    # Helper function to save plots to both reports and static directory
    def save_dual_plot(fig, filename):
        path1 = os.path.join(output_dir, filename)
        path2 = os.path.join(static_dir, filename)
        fig.savefig(path1, dpi=300, bbox_inches='tight')
        fig.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {path1} & {path2}")

    # =========================================================================
    # VISUALIZATION 1: Legitimate vs Fraudulent Transaction Count (Imbalance)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Legitimate (0)', 'Fraudulent (1)'], [legit_count, fraud_count],
                  color=['#2ecc71', '#e74c3c'], edgecolor='black', width=0.5)

    # Use logarithmic scale on y-axis so the 492 fraud cases are visible next to 284,315 cases
    ax.set_yscale('log')
    ax.set_title('Transaction Class Distribution (Log Scale)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Transactions (Logarithmic Scale)', fontsize=12)
    ax.set_xlabel('Transaction Class', fontsize=12)

    # Annotate bar heights with actual count and percentage
    for bar, count, pct in zip(bars, [legit_count, fraud_count], [legit_pct, fraud_pct]):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval * 1.3,
                f"{count:,}\n({pct:.3f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_dual_plot(fig, 'class_distribution.png')

    # =========================================================================
    # VISUALIZATION 2: Amount Distribution Comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Legitimate Amounts (filtered to <= 1000 for visibility)
    sns.histplot(amount_legit[amount_legit <= 1000], bins=50, kde=True, ax=axes[0], color='#2ecc71')
    axes[0].set_title('Legitimate Transaction Amounts (≤ €1,000)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Amount (€)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)

    # Fraudulent Amounts
    sns.histplot(amount_fraud, bins=30, kde=True, ax=axes[1], color='#e74c3c')
    axes[1].set_title('Fraudulent Transaction Amounts (Full Range)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Amount (€)', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)

    plt.tight_layout()
    save_dual_plot(fig, 'amount_distribution.png')

    # =========================================================================
    # VISUALIZATION 3: Fraud Amount Boxplot & Breakdown
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Class', y='Amount', data=df, ax=ax, hue='Class', palette=['#2ecc71', '#e74c3c'], showfliers=False, legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'], fontsize=11)
    ax.set_title('Transaction Amounts by Class (Outliers Excluded for Scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Amount (€)', fontsize=11)
    ax.set_xlabel('Transaction Class', fontsize=11)
    save_dual_plot(fig, 'fraud_amount_distribution.png')

    # =========================================================================
    # VISUALIZATION 4: Time Distribution Over 48 Hours
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    # Convert seconds to hours
    time_hours_legit = df[df['Class'] == 0]['Time'] / 3600
    time_hours_fraud = df[df['Class'] == 1]['Time'] / 3600

    sns.kdeplot(time_hours_legit, ax=ax, label='Legitimate', color='#2ecc71', fill=True, alpha=0.3)
    sns.kdeplot(time_hours_fraud, ax=ax, label='Fraudulent', color='#e74c3c', fill=True, alpha=0.3)

    ax.set_title('Transaction Density Over Time (Hours since start)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (Hours)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_xlim(0, 48)
    ax.legend(loc='upper right', frameon=True)
    save_dual_plot(fig, 'time_distribution.png')

    # =========================================================================
    # VISUALIZATION 5: Top Correlated Features with Fraud (Class)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    correlations = df.corr()['Class'].drop('Class').sort_values()
    # Pick top 6 negative and top 6 positive correlated features
    top_corr = pd.concat([correlations.head(6), correlations.tail(6)])

    colors = ['#e74c3c' if x < 0 else '#3498db' for x in top_corr.values]
    bars = ax.barh(top_corr.index, top_corr.values, color=colors, edgecolor='black')
    ax.set_title('Features Most Correlated with Fraud (Class)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Annotate bars
    for bar in bars:
        width = bar.get_width()
        offset = 0.01 if width >= 0 else -0.04
        ax.text(width + offset, bar.get_y() + bar.get_height()/2.0,
                f'{width:.2f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_dual_plot(fig, 'correlation_features.png')

    print("\nEDA Completed Successfully! All visualizations saved to:")
    print(f" - {output_dir}")
    print(f" - {static_dir}")
    print("=" * 70)

if __name__ == '__main__':
    run_eda()
