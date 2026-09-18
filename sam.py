# ============================================================
# TASK 1: RETAIL SALES DATA ANALYSIS & BUSINESS INSIGHTS
# ============================================================

# -----------------------------
# 1. Import Required Libraries
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 2. Load Dataset
# -----------------------------

file_name = "retail_sales.csv"

df = pd.read_csv(file_name)

print("\n========== ORIGINAL DATA ==========")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())


# -----------------------------
# 3. Check Missing Values
# -----------------------------

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)


# -----------------------------
# 4. Remove Duplicate Records
# -----------------------------

print("\n========== DUPLICATE RECORDS ==========")

print("Number of duplicate records:",
      df.duplicated().sum())

df = df.drop_duplicates()

print("Duplicates after removal:",
      df.duplicated().sum())


# -----------------------------
# 5. Clean Column Names
# -----------------------------

df.columns = df.columns.str.strip()

print("\nCleaned Columns:")
print(df.columns.tolist())


# -----------------------------
# 6. Handle Missing Values
# -----------------------------

# Numerical columns
numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )


# Categorical columns
categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# -----------------------------
# 7. Remove Extra Spaces
# -----------------------------

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()


# -----------------------------
# 8. Standardize Text Columns
# -----------------------------

# These columns are cleaned only if they exist.

if "Category" in df.columns:
    df["Category"] = df["Category"].str.title()

if "Region" in df.columns:
    df["Region"] = df["Region"].str.title()

if "Product" in df.columns:
    df["Product"] = df["Product"].str.title()


# -----------------------------
# 9. Convert Date Column
# -----------------------------

if "Order Date" in df.columns:

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

    # Remove rows with invalid dates
    df = df.dropna(subset=["Order Date"])


# -----------------------------
# 10. Convert Numerical Columns
# -----------------------------

for column in ["Sales", "Profit", "Quantity"]:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].fillna(
            df[column].median()
        )


# -----------------------------
# 11. Final Data Check
# -----------------------------

print("\n========== CLEANED DATA ==========")

print(df.head())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset shape after cleaning:")
print(df.shape)


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================


# -----------------------------
# 12. Statistical Summary
# -----------------------------

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())


# -----------------------------
# 13. Total Sales
# -----------------------------

if "Sales" in df.columns:

    total_sales = df["Sales"].sum()

    print("\nTotal Sales:",
          round(total_sales, 2))


# -----------------------------
# 14. Total Profit
# -----------------------------

if "Profit" in df.columns:

    total_profit = df["Profit"].sum()

    print("Total Profit:",
          round(total_profit, 2))


# -----------------------------
# 15. Total Quantity
# -----------------------------

if "Quantity" in df.columns:

    total_quantity = df["Quantity"].sum()

    print("Total Quantity Sold:",
          total_quantity)


# -----------------------------
# 16. Average Sales
# -----------------------------

if "Sales" in df.columns:

    average_sales = df["Sales"].mean()

    print("Average Sales per Order:",
          round(average_sales, 2))


# -----------------------------
# 17. Average Profit
# -----------------------------

if "Profit" in df.columns:

    average_profit = df["Profit"].mean()

    print("Average Profit per Order:",
          round(average_profit, 2))


# ============================================================
# CUSTOMER ANALYSIS
# ============================================================


# -----------------------------
# 18. Total Customers
# -----------------------------

if "Customer ID" in df.columns:

    total_customers = df["Customer ID"].nunique()

    print("\n========== CUSTOMER ANALYSIS ==========")

    print("Total Unique Customers:",
          total_customers)


# -----------------------------
# 19. Customer Sales
# -----------------------------

if "Customer Name" in df.columns and "Sales" in df.columns:

    customer_sales = (
        df.groupby("Customer Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 Customers by Sales:")

    print(customer_sales.head(10))


# -----------------------------
# 20. Customer Profit
# -----------------------------

if "Customer Name" in df.columns and "Profit" in df.columns:

    customer_profit = (
        df.groupby("Customer Name")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 Customers by Profit:")

    print(customer_profit.head(10))


# ============================================================
# PRODUCT ANALYSIS
# ============================================================


# -----------------------------
# 21. Product Sales
# -----------------------------

if "Product" in df.columns and "Sales" in df.columns:

    product_sales = (
        df.groupby("Product")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== PRODUCT ANALYSIS ==========")

    print("\nTop 10 Products by Sales:")

    print(product_sales.head(10))


# -----------------------------
# 22. Product Quantity
# -----------------------------

if "Product" in df.columns and "Quantity" in df.columns:

    product_quantity = (
        df.groupby("Product")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 Products by Quantity:")

    print(product_quantity.head(10))


# -----------------------------
# 23. Product Profit
# -----------------------------

if "Product" in df.columns and "Profit" in df.columns:

    product_profit = (
        df.groupby("Product")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nTop 10 Products by Profit:")

    print(product_profit.head(10))


# ============================================================
# CATEGORY ANALYSIS
# ============================================================


# -----------------------------
# 24. Category Sales
# -----------------------------

if "Category" in df.columns and "Sales" in df.columns:

    category_sales = (
        df.groupby("Category")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== CATEGORY ANALYSIS ==========")

    print("\nSales by Category:")

    print(category_sales)


# -----------------------------
# 25. Category Profit
# -----------------------------

if "Category" in df.columns and "Profit" in df.columns:

    category_profit = (
        df.groupby("Category")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nProfit by Category:")

    print(category_profit)


# -----------------------------
# 26. Category Quantity
# -----------------------------

if "Category" in df.columns and "Quantity" in df.columns:

    category_quantity = (
        df.groupby("Category")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nQuantity by Category:")

    print(category_quantity)


# ============================================================
# REGIONAL ANALYSIS
# ============================================================


# -----------------------------
# 27. Region Sales
# -----------------------------

if "Region" in df.columns and "Sales" in df.columns:

    region_sales = (
        df.groupby("Region")["Sales"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\n========== REGIONAL ANALYSIS ==========")

    print("\nSales by Region:")

    print(region_sales)


# -----------------------------
# 28. Region Profit
# -----------------------------

if "Region" in df.columns and "Profit" in df.columns:

    region_profit = (
        df.groupby("Region")["Profit"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nProfit by Region:")

    print(region_profit)


# -----------------------------
# 29. Region Quantity
# -----------------------------

if "Region" in df.columns and "Quantity" in df.columns:

    region_quantity = (
        df.groupby("Region")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nQuantity by Region:")

    print(region_quantity)


# ============================================================
# TIME SERIES ANALYSIS
# ============================================================


# -----------------------------
# 30. Monthly Sales
# -----------------------------

if "Order Date" in df.columns and "Sales" in df.columns:

    df["Month"] = df["Order Date"].dt.to_period("M")

    monthly_sales = (
        df.groupby("Month")["Sales"]
        .sum()
    )

    print("\n========== MONTHLY SALES ==========")

    print(monthly_sales)


# -----------------------------
# 31. Monthly Profit
# -----------------------------

if "Order Date" in df.columns and "Profit" in df.columns:

    monthly_profit = (
        df.groupby("Month")["Profit"]
        .sum()
    )

    print("\nMonthly Profit:")

    print(monthly_profit)


# ============================================================
# VISUALIZATIONS
# ============================================================


# -----------------------------
# 32. Bar Chart - Category Sales
# -----------------------------

if "Category" in df.columns and "Sales" in df.columns:

    plt.figure(figsize=(8, 5))

    category_sales.plot(
        kind="bar"
    )

    plt.title("Sales by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 33. Bar Chart - Region Sales
# -----------------------------

if "Region" in df.columns and "Sales" in df.columns:

    plt.figure(figsize=(8, 5))

    region_sales.plot(
        kind="bar"
    )

    plt.title("Sales by Region")
    plt.xlabel("Region")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 34. Top 10 Products Bar Chart
# -----------------------------

if "Product" in df.columns and "Sales" in df.columns:

    top_products = product_sales.head(10)

    plt.figure(figsize=(10, 6))

    top_products.plot(
        kind="bar"
    )

    plt.title("Top 10 Products by Sales")
    plt.xlabel("Product")
    plt.ylabel("Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 35. Monthly Sales Line Chart
# -----------------------------

if "Order Date" in df.columns and "Sales" in df.columns:

    plt.figure(figsize=(12, 6))

    monthly_sales.plot(
        kind="line",
        marker="o"
    )

    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Sales")

    plt.xticks(rotation=45)

    plt.grid(True)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 36. Monthly Profit Line Chart
# -----------------------------

if "Order Date" in df.columns and "Profit" in df.columns:

    plt.figure(figsize=(12, 6))

    monthly_profit.plot(
        kind="line",
        marker="o"
    )

    plt.title("Monthly Profit Trend")
    plt.xlabel("Month")
    plt.ylabel("Profit")

    plt.xticks(rotation=45)

    plt.grid(True)

    plt.tight_layout()
    plt.show()


# -----------------------------
# 37. Scatter Plot
# Quantity vs Sales
# -----------------------------

if "Quantity" in df.columns and "Sales" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="Quantity",
        y="Sales"
    )

    plt.title("Quantity vs Sales")
    plt.xlabel("Quantity")
    plt.ylabel("Sales")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 38. Scatter Plot
# Sales vs Profit
# -----------------------------

if "Sales" in df.columns and "Profit" in df.columns:

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x="Sales",
        y="Profit"
    )

    plt.title("Sales vs Profit")
    plt.xlabel("Sales")
    plt.ylabel("Profit")

    plt.tight_layout()
    plt.show()


# -----------------------------
# 39. Correlation Heatmap
# -----------------------------

numeric_columns_for_heatmap = [
    column for column in
    ["Quantity", "Sales", "Profit"]
    if column in df.columns
]

if len(numeric_columns_for_heatmap) >= 2:

    correlation = df[
        numeric_columns_for_heatmap
    ].corr()

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()


# ============================================================
# BUSINESS INSIGHTS
# ============================================================


print("\n")
print("=" * 60)
print("             BUSINESS INSIGHTS")
print("=" * 60)


# Best Product

if "Product" in df.columns and "Sales" in df.columns:

    best_product = product_sales.idxmax()

    best_product_sales = product_sales.max()

    print(
        "\n1. Best-selling Product:",
        best_product,
        "| Sales:",
        round(best_product_sales, 2)
    )


# Best Category

if "Category" in df.columns and "Sales" in df.columns:

    best_category = category_sales.idxmax()

    best_category_sales = category_sales.max()

    print(
        "\n2. Highest Sales Category:",
        best_category,
        "| Sales:",
        round(best_category_sales, 2)
    )


# Best Region

if "Region" in df.columns and "Sales" in df.columns:

    best_region = region_sales.idxmax()

    best_region_sales = region_sales.max()

    print(
        "\n3. Highest Sales Region:",
        best_region,
        "| Sales:",
        round(best_region_sales, 2)
    )


# Most Profitable Category

if "Category" in df.columns and "Profit" in df.columns:

    profitable_category = category_profit.idxmax()

    profitable_category_profit = category_profit.max()

    print(
        "\n4. Most Profitable Category:",
        profitable_category,
        "| Profit:",
        round(profitable_category_profit, 2)
    )


# Top Customer

if "Customer Name" in df.columns and "Sales" in df.columns:

    top_customer = customer_sales.idxmax()

    top_customer_sales = customer_sales.max()

    print(
        "\n5. Top Customer:",
        top_customer,
        "| Sales:",
        round(top_customer_sales, 2)
    )


# Highest Sales Month

if "Order Date" in df.columns and "Sales" in df.columns:

    highest_sales_month = monthly_sales.idxmax()

    highest_month_sales = monthly_sales.max()

    print(
        "\n6. Highest Sales Month:",
        highest_sales_month,
        "| Sales:",
        round(highest_month_sales, 2)
    )


print("\n")
print("=" * 60)
print("             ANALYSIS COMPLETED")
print("=" * 60)