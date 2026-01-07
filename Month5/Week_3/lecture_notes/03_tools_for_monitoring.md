Lecture 3: Open-Source Tools for Model Monitoring

While cloud platforms offer integrated monitoring solutions, a powerful and flexible stack can be built using open-source tools. This lecture provides an overview of a popular combination: Evidently AI, Prometheus, and Grafana.

 The Monitoring Stack: How They Work Together

These three tools form a complete monitoring loop: Evidently AI calculates the ML-specific metrics, Prometheus stores them, and Grafana visualizes them.

1.  Evidently AI (The ML Specialist): Calculates detailed metrics about data drift, concept drift, and model performance from your model's live data.
2.  Prometheus (The Time-Series Database): "Scrapes" (collects) these metrics from an endpoint exposed by your service and stores them as time-series data.
3.  Grafana (The Dashboard): Connects to Prometheus as a data source and builds interactive dashboards to visualize the metrics over time, and can be configured to send alerts.

![Monitoring Stack](https://raw.githubusercontent.com/evidentlyai/evidently/main/docs/book/assets/monitoring/grafana-dashboard-service.png)
Image source: Evidently AI documentation

---

 1. Evidently AI

Evidently AI is an open-source Python library specifically designed to evaluate, test, and monitor ML models.

- What it does: It compares two datasets—typically a reference dataset (like your training data) and a current dataset (live production data)—and generates detailed reports and metrics.
- Key Features:
    - Comprehensive Reports: Can generate interactive HTML reports showing data drift, target drift, data quality, and model performance.
    - Metrics Presets: Provides pre-built sets of metrics for common tasks (e.g., `DataDriftPreset`, `ClassificationPreset`).
    - JSON Profiles: Can output results as a JSON object, which is perfect for integration with other systems.
    - Monitoring Service: Can be deployed as a long-running service that continuously computes metrics and exposes them for Prometheus to scrape.

 How it Works
You would typically create a separate Python service that runs alongside your main model API. This monitoring service periodically fetches the latest production data, compares it to a reference dataset using Evidently, and calculates the necessary metrics.

---

 2. Prometheus

Prometheus is a leading open-source monitoring and alerting system, originally built at SoundCloud. It is now a standard tool for monitoring infrastructure and applications.

- What it does: It collects and stores metrics as time-series data.
- Data Model: Every time series is identified by a metric name and a set of key-value pairs (labels).
- Pull-based Model: Prometheus works by "scraping" HTTP endpoints on the services it monitors. Your monitoring service must expose its metrics (e.g., "data_drift_score") on an endpoint like `/metrics`.
- PromQL: Prometheus has a powerful query language called PromQL that allows you to select and aggregate time-series data in real time.

---

 3. Grafana

Grafana is an open-source analytics and interactive visualization web application. It allows you to create beautiful and complex dashboards from your time-series data.

- What it does: It visualizes data from a wide variety of data sources, including Prometheus.
- Dashboards: You can build dashboards with different types of panels (graphs, tables, gauges, heatmaps) to track your model's health. For ML monitoring, you might create panels for:
    - Model accuracy over time.
    - Data drift score for key features.
    - The distribution of your model's predictions.
    - API latency and request count.
- Alerting: Grafana has a powerful alerting engine. You can set up rules to automatically send notifications (e.g., to Slack or email) if a metric crosses a certain threshold (e.g., if `data_drift_score > 0.2` or `model_accuracy < 0.85`). This allows you to be proactive about model maintenance.

 Summary of the Workflow

1.  Your ML model serves predictions and logs the input features and final predictions to a database or log file.
2.  An Evidently AI service runs periodically (e.g., every hour). It reads the recent production data and a reference dataset.
3.  The service uses Evidently to compute drift scores and performance metrics.
4.  The service exposes these metrics on an HTTP endpoint (e.g., `/metrics`).
5.  Prometheus is configured to scrape this endpoint at regular intervals, collecting and storing the metrics.
6.  Grafana is connected to Prometheus as a data source. You and your team view the dashboards in Grafana to observe model behavior.
7.  Grafana's alerting rules notify you if any metric indicates a potential problem, signaling that it might be time to investigate or retrain the model.