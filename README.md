#  SME Demand Forecasting & Reorder Optimization Engine

A Streamlit-based application that helps small and medium-sized businesses forecast product demand, optimize inventory, and make data-driven replenishment decisions.

---

##  Overview

This project uses machine learning (LightGBM) to forecast SKU-level demand based on historical sales data. It automatically processes Excel uploads, engineers time-based features, and generates actionable insights such as forecast quantities, safety stock, reorder recommendations, and forecast accuracy.

---

##  Key Features

* **Automated Data Processing**

  * Detects SKU, Date, and Sales columns
  * Cleans and normalizes messy Excel data

* 🤖 **Machine Learning Forecasting**

  * Uses LightGBM regression model
  * Applies lag features and seasonality signals

* **Inventory Optimization**

  * Forecast demand per SKU
  * Calculates safety stock
  * Suggests reorder quantities

*  **Accuracy Tracking**

  * SKU-level forecast accuracy (%)
  * Highlights low-performing SKUs

*  **Interactive Dashboard**

  * KPI metrics (forecast, accuracy, SKU counts)
  * Visualizations using Altair
  * SKU-level drill-down analysis

*  **Export Results**

  * Download forecast results as Excel

---

##  How It Works

1. Upload historical sales data in Excel format
2. System automatically:

   * Cleans and validates data
   * Generates time-based and lag features
3. Trains a LightGBM model
4. Produces:

   * Demand forecast
   * Safety stock
   * Reorder recommendations
   * Accuracy metrics
5. Displays results in an interactive dashboard

---

##  Input Requirements

Your Excel file must contain at least:

* **SKU** (Product identifier)
* **Date** (Transaction date)
* **Sales** (Units sold)

Column names can vary (e.g., "product", "quantity") — the system auto-detects them.

---

##  Tech Stack

* Python
* Streamlit
* Pandas & NumPy
* LightGBM
* Altair

---

##  How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

##  Output Example

For each SKU, the system generates:

* Forecasted demand
* Safety stock level
* Recommended reorder quantity
* Forecast accuracy (%)

---

##  Use Cases

* Retail inventory planning
* Supply chain optimization
* SME demand forecasting
* Stock replenishment decisions

---

##  Limitations

* Requires sufficient historical data per SKU
* Forecast accuracy may vary for low-volume items
* Assumes consistent historical patterns

---

##  Future Improvements

* Multi-seasonality modeling
* External factors (promotions, holidays)
* Multi-location inventory support
* Real-time API integration

---

##  License

MIT License

