# Movie Recommendation System Project

## Overview
In this project, you'll build a movie recommendation system using different techniques:
- Content-based filtering
- Collaborative filtering
- Hybrid approach

## Dataset
We'll use the [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/), which contains:
- 25 million ratings
- 1 million users
- 62,000 movies
- Genre information
- Tag information

## Project Structure
```
project/
├── data/                    # Raw and processed data
├── notebooks/               # Jupyter notebooks for exploration
│   ├── 1_exploratory_analysis.ipynb
│   └── 2_model_experiments.ipynb
├── src/                     # Source code
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── models/
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   └── hybrid.py
│   └── evaluation.py
├── requirements.txt         # Project dependencies
└── README.md
```

## Tasks
1. **Data Exploration** (2 hours)
   - Load and explore the dataset
   - Visualize rating distributions
   - Analyze user behavior patterns

2. **Content-Based Filtering** (3 hours)
   - Implement TF-IDF for movie descriptions
   - Calculate cosine similarity between movies
   - Build a content-based recommender

3. **Collaborative Filtering** (3 hours)
   - Implement user-based and item-based filtering
   - Use matrix factorization (SVD)
   - Evaluate model performance

4. **Hybrid Approach** (2 hours)
   - Combine content-based and collaborative filtering
   - Implement weighted hybrid recommendations

5. **Evaluation** (2 hours)
   - Use RMSE and MAE for rating prediction
   - Measure recommendation diversity
   - Conduct user studies

## Deliverables
1. Jupyter notebooks with analysis and visualizations
2. Python modules with reusable components
3. Model evaluation report
4. Presentation slides (5-7 minutes)

## Timeline
- **Day 1-2**: Data exploration and preprocessing
- **Day 3-4**: Content-based filtering implementation
- **Day 5-6**: Collaborative filtering implementation
- **Day 7**: Hybrid approach and final evaluation

## Getting Started
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset and extract it to `data/raw/`
4. Run the notebooks in sequence

## Evaluation Criteria
- Code quality and organization (20%)
- Model performance (30%)
- Analysis and insights (20%)
- Documentation (15%)
- Presentation (15%)
