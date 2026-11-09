# Creates a deliberately messy "raw" copy of the survey so the cleaning step has real work to do.
# Run once:  python make_raw_data.py   (seed fixed, so results are repeatable)
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
d = pd.read_csv('data/trek_care_survey_dataset.csv'); n = len(d)
ATT = ['q14_novelty','q15_natural','q16_plastic_free','q17_himalayan_origin','q18_return_tin','q19_color_code','q20_trust_local','q21_gifts','q22_price_priority']
vary = lambda x: rng.choice([x.lower(), x.upper(), ' ' + x, x + ' '])
for c in ['q3_country', 'q4_traveler_type']:                      # 1. inconsistent labels
    i = rng.choice(n, int(.08 * n), replace=False); d.loc[i, c] = [vary(v) for v in d.loc[i, c]]
d.loc[rng.choice(d.index[d.q3_country.eq('UAE')], 60, replace=False), 'q3_country'] = 'U.A.E'
for c in ['q5_income_band', 'q11_severity', 'q13_past_spend_band'] + ATT:   # 2. missing answers (3%)
    d.loc[rng.choice(n, int(.03 * n), replace=False), c] = np.nan
for _ in range(30): d.loc[rng.integers(n), rng.choice(ATT)] = rng.choice([0, 6])   # 3. impossible Likert values
d['q28_currency'] = 'NPR'                                         # 4. some UAE respondents typed spend in AED
i = rng.choice(d.index[d.q3_country.str.strip().str.lower().isin(['uae','u.a.e']) & (d.q28_annual_spend_npr > 0)], 120, replace=False)
d.loc[i, 'q28_currency'] = 'AED'; d.loc[i, 'q28_annual_spend_npr'] = (d.loc[i, 'q28_annual_spend_npr'] / 36.2).round(0)
j = rng.choice(d.index[(d.q28_annual_spend_npr > 0) & (d.q28_currency == 'NPR')], 15, replace=False)
d.loc[j, 'q28_annual_spend_npr'] *= 40                            # 5. extreme outliers
d = pd.concat([d, d.sample(60, random_state=7)])                  # 6. duplicate respondents
d.to_csv('data/trek_care_raw_messy.csv', index=False); print(d.shape)
