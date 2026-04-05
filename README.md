# 🎬 Movie Recommendation System

This is an experimental project focused on technical understanding of practical implementation of multiple recommendation strategies
---

## Features

- Three recommendation strategies in one app
- Streamlit interface with mode selection in the sidebar
- Object-oriented code organization for easier maintenance

---

## Preview

### Popularity-Based mode

![Popularity-Based mode](assets/screenshots/popularity_mode.png)

### Content-Based mode

![Content-Based mode](assets/screenshots/content_mode.png)

### Collaborative Filtering mode

![Collaborative Filtering mode](assets/screenshots/collaborative_mode.png)

---

## Project Structure

```bash
movie_recommender_portfolio/
├── app/
│   └── main.py
├── assets/
│   └── screenshots/
├── data/
│   ├── credits.csv
│   ├── movies.csv
│   └── ratings.csv
├── notebooks/
│   ├── content_based_filtering.ipynb
│   ├── ML Collaborative Filtering.ipynb
│   └── movierec.ipynb
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── models/
│   │   ├── collaborative_recommender.py
│   │   ├── content_recommender.py
│   │   └── popularity_recommender.py
│   └── services/
│       └── recommendation_service.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Implemented Approaches

### 1. Popularity-Based Recommendation
This approach recommends movies based on weighted popularity scores. It is useful for suggesting broadly well-rated titles, especially when no user-specific preference history is available.

### 2. Content-Based Recommendation
This approach uses TF-IDF vectorization on movie overviews and computes similarity between titles based on textual content. It is designed to recommend movies that are similar in theme or description to a selected title.

### 3. Collaborative Filtering
This approach uses SVD from the Surprise library to estimate user preferences based on interaction patterns. It aims to generate more personalized recommendations by learning from user-item relationships.


---

## Technologies Used

- Python **3.12** ⚠️
- Streamlit
- pandas
- scikit-learn
- scikit-surprise

---

## How to Run

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app/main.py
```

---

## Notes

- The collaborative filtering mode depends on **scikit-surprise**.
- The original notebooks I used while studying were preserved in the `notebooks/` folder for reference and comparison.

---
