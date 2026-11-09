# Trek Care analytics dashboard

Natural, sustainably packaged personal care for trekkers (made in Nepal, exported to the UAE).
Streamlit dashboard: data cleaning, descriptive and diagnostic EDA, four algorithms (classification, clustering,
association rules, regression), and a feasibility verdict.

## Files
- `app.py` - the dashboard (all analysis code)
- `make_raw_data.py` - creates the deliberately messy raw file from the clean one (seed 7)
- `data/trek_care_survey_dataset.csv` - clean synthetic survey, 4,000 respondents (money in NPR)
- `data/trek_care_raw_messy.csv` - raw file with duplicates, label variants, missing values, impossible scores, AED entries, outliers
- `requirements.txt` - Python packages

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Data is synthetic (generated with Python assumptions), not real survey responses.
- Money is stored in NPR; the sidebar can show AED (1 AED = 36.2 NPR, assumed).
- Page 5 tunes 8 classifiers, so its first load takes a few minutes. The result is then cached.
