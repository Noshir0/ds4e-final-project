# NYC Taxi Fare Prediction

Final project for DS-UA 9111 (Data Science for Everyone), NYU.

This is a Streamlit app that predicts how much a New York City taxi trip will
cost, using linear regression. The basic question was: can you estimate the
fare from simple trip details like distance, time, and where you got picked up?

## Data

I used a sample of NYC taxi trips (about 6,300 rows after removing bad records
like zero-distance or impossibly long trips). The columns I used:

- distance, trip duration, number of passengers, pickup hour
- day of week, payment type, taxi color, pickup borough, dropoff borough

The target is the total cost of the trip in dollars. I dropped the `fare`,
`tip`, and `tolls` columns on purpose, because together they add up to the
total, so using them would basically be giving the model the answer.

## Pages

The app has 6 pages, switchable from the sidebar:

1. Business case and data overview
2. Data visualization and a few findings
3. Prediction - you can switch between Linear Regression, Ridge, and Random Forest
4. Explainable AI - SHAP feature importance
5. Hyperparameter tuning - tracked with Weights & Biases
6. Conclusion

## A few findings

- Distance and trip duration explain almost all of the cost. Number of
  passengers basically doesn't matter, since fares are charged per trip, not
  per person.
- Linear regression already gets about 93% R2. A tuned random forest gets to
  about 94%, mostly by picking up the JFK airport flat fare.

## How to run

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files

- `streamlit_app.py` - the app (all 6 pages)
- `nyc_taxis.csv` - the dataset
- `requirements.txt` - dependencies

## Live versions

- Streamlit Cloud: https://ds4e-final-nyc-taxi-fare.streamlit.app/
- Hugging Face: https://huggingface.co/spaces/Jonathan0123/nyc-taxi-fare-lab
