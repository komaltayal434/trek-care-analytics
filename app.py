"""Trek Care analytics dashboard (Streamlit).
Flow: raw messy survey -> cleaning and transformation -> descriptive and diagnostic EDA
      -> 4 algorithms (classification, clustering, association rules, regression) -> feasibility verdict.
Money is stored in NPR (home currency of the business). The sidebar can show it in AED (1 AED = 36.2 NPR, assumed)."""
import itertools
from pathlib import Path
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, silhouette_score, r2_score, mean_absolute_error, mean_squared_error)

st.set_page_config(page_title="Trek Care Analytics", layout="wide")
BASE = Path(__file__).parent
AED = 36.2  # assumed NPR per 1 AED
CTRY = {'nepal': 'Nepal', 'uae': 'UAE', 'europe': 'Europe', 'north america': 'North America', 'australia/nz': 'Australia/NZ',
        'india': 'India', 'other asia': 'Other Asia', 'other': 'Other'}
TYPE = {'international trekker': 'International trekker', 'nepal resident': 'Nepal resident',
        'uae resident': 'UAE resident', 'nepali diaspora': 'Nepali diaspora'}
ATT = ['q14_novelty', 'q15_natural', 'q16_plastic_free', 'q17_himalayan_origin', 'q18_return_tin', 'q19_color_code',
       'q20_trust_local', 'q21_gifts', 'q22_price_priority']
PROB = ['q10_chapped_lips', 'q10_sunburn', 'q10_windburn_dry_nose', 'q10_cracked_heels', 'q10_chafing',
        'q10_insect_bites', 'q10_muscle_soreness', 'q10_cracked_hands']
INC = {'NPR': ['<65k', '65-200k', '200-400k', '400-800k', '800k+'], 'AED': ['<2k', '2-5.5k', '5.5-11k', '11-22k', '22k+']}
AGE = ['18-24', '25-34', '35-44', '45-54', '55+']
MONEY = ['spend', 'p_too_cheap', 'p_bargain', 'p_expensive', 'p_too_expensive']


# ---------------------------------------------------------------- 1. LOAD, CLEAN, TRANSFORM
@st.cache_data
def load():
    raw = pd.read_csv(BASE / 'data' / 'trek_care_raw_messy.csv')
    d, L = raw.copy(), []
    def log(issue, how, fix, why, n): L.append({'Issue': issue, 'How detected': how, 'Treatment': fix, 'Why': why, 'Count': int(n)})
    n = d.duplicated().sum(); d = d.drop_duplicates().reset_index(drop=True)
    log('Duplicate respondents', 'df.duplicated()', 'Dropped repeated rows', 'A person counted twice distorts every percentage', n)
    for c, m in (('q3_country', CTRY), ('q4_traveler_type', TYPE)):
        new = d[c].astype(str).str.strip().str.lower().str.replace('.', '', regex=False).map(m)
        n = (new != d[c]).sum(); d[c] = new
        log(f'Inconsistent labels in {c}', 'value_counts()', 'Trim spaces, lower-case, remove dots, map to one label', 'One category needs one name', n)
    bad = 0
    for c in ATT:
        m = d[c].notna() & ~d[c].between(1, 5); bad += m.sum(); d.loc[m, c] = np.nan
    log('Likert scores outside 1-5', 'describe(): min and max', 'Set to missing, then imputed', 'The scale only allows 1 to 5', bad)
    m = d.q28_currency.eq('AED')
    d.loc[m, 'q28_annual_spend_npr'] = (d.loc[m, 'q28_annual_spend_npr'] * AED / 10).round() * 10
    d = d.drop(columns='q28_currency')
    log('Spend typed in AED', 'Currency column', f'Multiplied by {AED} NPR per AED', 'Mixed currencies make spend meaningless', m.sum())
    b = d.q28_annual_spend_npr[d.q28_annual_spend_npr > 0]; q1, q3 = b.quantile([.25, .75]); cap = round(q3 + 3 * (q3 - q1), -1)
    m = d.q28_annual_spend_npr > cap; d.loc[m, 'q28_annual_spend_npr'] = cap
    log('Extreme spend outliers', f'IQR rule: above Q3 + 3 x IQR = NPR {cap:,.0f}', 'Capped at that fence (winsorizing)', 'A few extreme values distort means and regression', m.sum())
    miss = d.isna().sum(); miss = miss[miss > 0]
    for c in miss.index: d[c] = d[c].fillna(d[c].median()).round().astype(int)
    log('Missing answers', 'isna().sum()', 'Filled with the column median', 'Ordinal scales: the median keeps the distribution', miss.sum())
    bask = [c for c in d if c.startswith('q25_')]
    d['n_problems'] = d[PROB].sum(axis=1); d['n_products_wanted'] = d[bask].sum(axis=1)
    d['interested_flag'] = (d.q24_likelihood >= 4).astype(int)
    d['age_code'] = d.q1_age_group.map({a: i for i, a in enumerate(AGE)})
    d['eco_score'] = d[['q15_natural', 'q16_plastic_free', 'q17_himalayan_origin', 'q18_return_tin']].mean(axis=1).round(2)
    d['trek_intensity'] = d.q6_treks_per_year + d.q7_trip_length + d.q8_altitude
    d = d.rename(columns={'q28_annual_spend_npr': 'spend', 'q26_too_cheap_npr': 'p_too_cheap', 'q26_bargain_npr': 'p_bargain',
                          'q26_expensive_npr': 'p_expensive', 'q26_too_expensive_npr': 'p_too_expensive'})
    return raw, d, pd.DataFrame(L), bask


RAW, D0, LOG, BASK = load()


def X_of(d):
    num = ['age_code', 'q5_income_band', 'q6_treks_per_year', 'q7_trip_length', 'q8_altitude', 'q9_plan_nepal_trek'] + PROB + \
          ['q11_severity', 'q12_n_products_carried', 'q13_past_spend_band'] + ATT + ['n_problems']
    return pd.concat([d[num], pd.get_dummies(d[['q2_gender', 'q3_country', 'q4_traveler_type']], drop_first=True).astype(int)], axis=1)


# ---------------------------------------------------------------- SIDEBAR
st.sidebar.title("Trek Care analytics")
PAGES = ['1. Overview', '2. Data cleaning', '3. Descriptive EDA', '4. Correlation and diagnostic EDA', '5. Classification',
         '6. Clustering', '7. Association rules', '8. Regression', '9. Feasibility verdict']
page = st.sidebar.radio("Go to", PAGES)
CUR = st.sidebar.radio("Show money in", ['NPR', 'AED'], horizontal=True)
FX = 1 / AED if CUR == 'AED' else 1.0
st.sidebar.caption(f"Stored in NPR. 1 AED = {AED} NPR (assumed rate).")
D = D0.copy(); D[MONEY] = D[MONEY] * FX            # D: full data in the chosen currency (used by models)
TYPES = st.sidebar.multiselect("Filter EDA pages by traveler type", sorted(D.q4_traveler_type.unique()), default=sorted(D.q4_traveler_type.unique()))
F = D[D.q4_traveler_type.isin(TYPES)]              # F: filtered data (used by EDA pages only)
money = lambda v: f"{CUR} {v:,.0f}"


def note(shows, why, insight, decision):
    st.info(f"**What it shows:** {shows}  \n**Why this chart:** {why}  \n**Insight:** {insight}  \n**Business decision:** {decision}")


def show(fig, **kw):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fig, width='stretch', **kw)


def why_box(text): st.success(text)


def mynotes(key): st.text_area("My notes: what I understood (write in your own words)", key=key, height=120)


# ---------------------------------------------------------------- MODELS (cached)
def models():
    mk = lambda m: Pipeline([('s', StandardScaler()), ('m', m)])
    return {
        'Logistic Regression': (mk(LogisticRegression(max_iter=3000)), {'m__C': [.01, .1, 1, 10], 'm__penalty': ['l1', 'l2'], 'm__solver': ['liblinear']}),
        'KNN': (mk(KNeighborsClassifier()), {'m__n_neighbors': [5, 11, 21, 31, 51], 'm__weights': ['uniform', 'distance']}),
        'Decision Tree': (DecisionTreeClassifier(random_state=1), {'max_depth': [3, 5, 7, 10], 'min_samples_leaf': [5, 20, 50], 'criterion': ['gini', 'entropy']}),
        'Random Forest': (RandomForestClassifier(random_state=1), {'n_estimators': [200, 400], 'max_depth': [6, 10, None], 'min_samples_leaf': [1, 5]}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=1), {'n_estimators': [100, 200], 'learning_rate': [.03, .1], 'max_depth': [2, 3]}),
        'SVM': (mk(SVC()), {'m__C': [.1, 1, 10], 'm__kernel': ['linear', 'rbf'], 'm__gamma': ['scale', .01]}),
        'Naive Bayes': (GaussianNB(), {'var_smoothing': np.logspace(-9, -3, 7)}),
        'AdaBoost': (AdaBoostClassifier(random_state=1), {'n_estimators': [100, 200, 400], 'learning_rate': [.1, .5, 1]})}


@st.cache_resource(show_spinner="Tuning 8 classifiers with 5-fold cross-validation. The first run takes 1 to 3 minutes...")
def run_clf():
    X, y = X_of(D0), D0.interested_flag
    Xa, Xt, ya, yt = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    rows, roc, cm, fitted = [], {}, {}, {}
    for n, (m, g) in models().items():
        base = accuracy_score(yt, m.fit(Xa, ya).predict(Xt))                       # default settings
        gs = GridSearchCV(m, g, cv=5, scoring='accuracy', n_jobs=-1).fit(Xa, ya)  # tuning on TRAIN data only
        b = gs.best_estimator_; p = b.predict(Xt)
        s = b.predict_proba(Xt)[:, 1] if hasattr(b, 'predict_proba') else b.decision_function(Xt)
        f, t, _ = roc_curve(yt, s); roc[n] = (f, t); cm[n] = confusion_matrix(yt, p); fitted[n] = b
        rows.append({'Model': n, 'Default accuracy': base, 'CV accuracy (tuned)': gs.best_score_, 'Test accuracy': accuracy_score(yt, p),
                     'Precision': precision_score(yt, p), 'Recall': recall_score(yt, p), 'F1': f1_score(yt, p), 'AUC': roc_auc_score(yt, s),
                     'Best parameters': str({k.split('__')[-1]: v for k, v in gs.best_params_.items()})})
    t = pd.DataFrame(rows).sort_values('Test accuracy', ascending=False).reset_index(drop=True)
    pm = [n for n in t.Model if hasattr(fitted[n], 'predict_proba')][0]
    pall = pd.Series(cross_val_predict(fitted[pm], X, y, cv=5, method='predict_proba')[:, 1], index=D0.index)
    return dict(table=t, roc=roc, cm=cm, pall=pall, pm=pm, ntr=len(Xa), nte=len(Xt))


CORE = ['q6_treks_per_year', 'q7_trip_length', 'q8_altitude', 'q9_plan_nepal_trek', 'q11_severity', 'q14_novelty', 'q15_natural',
        'q16_plastic_free', 'q17_himalayan_origin', 'q21_gifts', 'q22_price_priority', 'q5_income_band', 'n_problems']


@st.cache_data
def clu_scores():
    Z = StandardScaler().fit_transform(D0[CORE]); rows = []
    for k in range(2, 9):
        for ini in ('k-means++', 'random'):
            l = KMeans(k, init=ini, n_init=10, random_state=1).fit(Z)
            rows.append({'k': k, 'init': ini, 'silhouette': silhouette_score(Z, l.labels_), 'inertia': l.inertia_})
    return pd.DataFrame(rows)


@st.cache_data
def clu_labels(k):
    Z = StandardScaler().fit_transform(D0[CORE]); l = KMeans(k, n_init=20, random_state=1).fit_predict(Z)
    order = pd.Series(D0.interested_flag.values).groupby(l).mean().sort_values().index.tolist()
    names = ['Less probable', 'Medium probable', 'Most probable'] if k == 3 else [f'Segment {i + 1} (low to high interest)' for i in range(k)]
    return pd.Series([names[order.index(x)] for x in l], index=D0.index), PCA(2, random_state=1).fit_transform(Z)


@st.cache_data
def supports():
    B = D0[BASK].values.astype(bool)
    return {it: B[:, list(it)].all(axis=1).mean() for s in (1, 2, 3) for it in itertools.combinations(range(len(BASK)), s)}


def rules(S, ms, mc, ml):
    nm, out = [c[4:] for c in BASK], []
    for it, s in S.items():
        if len(it) < 2 or s < ms: continue
        for r in range(1, len(it)):
            for a in itertools.combinations(it, r):
                b = tuple(x for x in it if x not in a); conf = s / S[a]; lift = conf / S[b]
                if conf >= mc and lift >= ml: out.append({'If customer wants': ' + '.join(nm[i] for i in a), 'Then also wants': ' + '.join(nm[i] for i in b),
                                                         'Support': s, 'Confidence': conf, 'Lift': lift})
    return pd.DataFrame(out)


@st.cache_data
def rule_grid():
    S, rows = supports(), []
    for ms in (.05, .1, .15, .2, .3):
        for mc in (.5, .6, .7, .8):
            r = rules(S, ms, mc, 1.1); rows.append({'min support': ms, 'min confidence': mc, 'rules found': len(r), 'mean lift': r.Lift.mean() if len(r) else 0})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Tuning Linear, Ridge and Lasso regression...")
def run_reg():
    X = X_of(D0).assign(q24_likelihood=D0.q24_likelihood); m = D0.spend > 0; Xb, yb = X[m], np.log1p(D0.spend[m])
    Xa, Xt, ya, yt = train_test_split(Xb, yb, test_size=.2, random_state=42); rows, fitted = [], {}
    mk = lambda e: Pipeline([('s', StandardScaler()), ('m', e)])
    for n, e, g in [('Linear', LinearRegression(), {'m__fit_intercept': [True]}), ('Ridge', Ridge(), {'m__alpha': np.logspace(-3, 3, 25)}),
                    ('Lasso', Lasso(max_iter=20000), {'m__alpha': np.logspace(-4, 0, 25)})]:
        gs = GridSearchCV(mk(e), g, cv=5, scoring='r2').fit(Xa, ya); p = gs.predict(Xt); po, yo = np.expm1(p), np.expm1(yt); fitted[n] = gs.best_estimator_
        rows.append({'Model': n, 'Best alpha': gs.best_params_.get('m__alpha', np.nan), 'CV R2 (log)': gs.best_score_, 'Test R2 (log)': r2_score(yt, p),
                     'Test R2 (NPR)': r2_score(yo, po), 'MAE (NPR)': mean_absolute_error(yo, po), 'RMSE (NPR)': mean_squared_error(yo, po) ** .5,
                     'Features dropped': int((gs.best_estimator_[-1].coef_ == 0).sum())})
    t = pd.DataFrame(rows); best = t.sort_values('Test R2 (log)', ascending=False).Model.iloc[0]
    coef = pd.Series(fitted['Lasso'][-1].coef_, index=X.columns)
    return dict(table=t, best=best, spend_all=pd.Series(np.expm1(fitted[best].predict(X)), index=D0.index), coef=coef,
                avp=pd.DataFrame({'Actual': np.expm1(yt), 'Predicted': np.expm1(fitted[best].predict(Xt))}), nb=int(m.sum()))


# ================================================================= PAGES
if page == PAGES[0]:
    st.title("Trek Care: demand and feasibility analytics")
    st.write("**Business idea:** a natural, sustainably packaged personal care line for trekkers, made in Nepal and exported to the UAE. "
             "**Question:** is there enough demand, at a workable price, to build a sustainable business?")
    st.write("**Data:** 4,000 synthetic survey respondents (generated with Python assumptions, not real people). The raw file is deliberately messy so the cleaning step is real.")
    c = st.columns(4); c[0].metric("Raw rows", f"{len(RAW):,}"); c[1].metric("Clean rows", f"{len(D0):,}")
    c[2].metric("Interested (Q24 is 4 or 5)", f"{D0.interested_flag.mean():.0%}"); c[3].metric("Avg spend if interested", money(D.loc[D.interested_flag == 1, 'spend'].mean()))
    st.subheader("Where each analytics type is used")
    st.table(pd.DataFrame({'Type': ['Descriptive', 'Diagnostic', 'Predictive', 'Prescriptive'],
                           'Question': ['What does the sample look like?', 'Why are some people more interested?', 'Who will buy and how much?', 'What should we do and is it viable?'],
                           'Pages': ['3', '4', '5, 6, 7, 8', '9']}))
    st.subheader("Raw data preview"); st.dataframe(RAW.head(10))

elif page == PAGES[1]:
    st.header("Data cleaning and transformation")
    st.write("The raw file has been checked column by column. Every issue, how it was found, what was done and why is logged below.")
    c = st.columns(3); c[0].metric("Raw shape", f"{RAW.shape[0]} x {RAW.shape[1]}"); c[1].metric("Clean shape", f"{D0.shape[0]} x {D0.shape[1]}"); c[2].metric("Raw missing cells", int(RAW.isna().sum().sum()))
    st.dataframe(LOG, width='stretch')
    st.subheader("Before and after")
    a, b = st.columns(2)
    with a:
        miss = RAW.isna().sum(); miss = miss[miss > 0].sort_values()
        show(px.bar(miss, orientation='h', title="Missing values in the raw file"))
    with b:
        both = pd.DataFrame({'Raw (NPR, mixed with AED)': RAW.q28_annual_spend_npr.describe(), 'Clean (NPR)': D0.spend.describe()}).round(0)
        st.write("Annual spend summary, raw vs clean"); st.dataframe(both)
    st.subheader("Transformations added")
    st.table(pd.DataFrame({'New column': ['n_problems', 'n_products_wanted', 'interested_flag', 'age_code', 'eco_score', 'trek_intensity'],
                           'How': ['Sum of the 8 problem flags', 'Sum of the 12 product yes/no answers', '1 if likelihood to buy is 4 or 5', 'Age group as 0 to 4',
                                   'Mean of natural, plastic-free, Himalayan, return-tin answers', 'Treks per year + trip length + altitude'],
                           'Why': ['Single measure of need', 'Basket size', 'Classification target', 'Numeric input for charts', 'One eco-attitude index', 'One exposure index']}))
    mynotes("n_clean")

elif page == PAGES[2]:
    st.header("Descriptive EDA: what does the sample look like?")
    if F.empty: st.warning("Pick at least one traveler type in the sidebar."); st.stop()
    g = F.groupby('q4_traveler_type').agg(Respondents=('respondent_id', 'count'), Interested=('interested_flag', 'mean')).reset_index()
    a, b = st.columns(2)
    with a:
        show(px.bar(g, x='q4_traveler_type', y='Respondents', text='Respondents', title="Respondents by traveler type"))
        note("Sample size per traveler type.", "A bar chart compares category counts clearly.", f"The largest group is {g.sort_values('Respondents').iloc[-1].q4_traveler_type}; interest ranges from {g.Interested.min():.0%} to {g.Interested.max():.0%}.", "Check that no segment is too small to trust before comparing them.")
    with b:
        show(px.bar(g, x='q4_traveler_type', y='Interested', text=g.Interested.map('{:.0%}'.format), title="Share interested by traveler type"))
        note("Top-2-box interest (likelihood 4 or 5) per segment.", "Rates, not counts, because segments differ in size.", f"{g.sort_values('Interested').iloc[-1].q4_traveler_type} are the most interested.", "Launch first where interest is highest.")
    a, b = st.columns(2)
    with a:
        ag = F.q1_age_group.value_counts().reindex(AGE); show(px.bar(ag, title="Age group", labels={'value': 'Respondents', 'index': 'Age'}))
        note("Age distribution of respondents.", "Ordered bars show how age is spread.", f"Most respondents are {ag.idxmax()}.", "Aim marketing at the dominant age bands.")
    with b:
        ic = F.q5_income_band.value_counts().sort_index(); ic.index = [INC[CUR][i - 1] for i in ic.index]
        show(px.bar(ic, title=f"Monthly household income ({CUR})", labels={'value': 'Respondents', 'index': 'Income band'}))
        note("Spread of household income.", "Ordered bars suit an ordinal scale.", f"The most common band is {ic.idxmax()}.", "Sets the price tier the line can realistically target.")
    a, b = st.columns(2)
    with a:
        pr = F[PROB].mean().sort_values(); pr.index = [i[4:].replace('_', ' ') for i in pr.index]
        show(px.bar(pr, orientation='h', title="Problems faced outdoors", labels={'value': 'Share of respondents', 'index': ''}))
        note("How common each skin problem is.", "A sorted horizontal bar ranks the problems.", f"'{pr.idxmax()}' is the most common ({pr.max():.0%}).", "Lead the range with the products that solve the most common problems.")
    with b:
        bu = F[F.spend > 0]; show(px.histogram(bu, x='spend', nbins=40, title=f"Expected annual spend of likely buyers ({CUR})"))
        note("Distribution of stated annual spend (zeros excluded).", "A histogram shows shape and skew.", f"Mean {money(bu.spend.mean())} is above median {money(bu.spend.median())}, so a few big spenders pull the mean up (right skew).", "Plan revenue on the median, and treat big spenders as upside.")
    a, b = st.columns(2)
    with a:
        lk = F.q24_likelihood.value_counts().sort_index(); show(px.bar(lk, title="Likelihood to buy (1 to 5)", labels={'value': 'Respondents', 'index': 'Score'}))
        note("Answers to the main purchase-intent question.", "Bars show the full scale, not just an average.", f"{(F.q24_likelihood >= 4).mean():.0%} answered 4 or 5.", "This top-2-box share is the demand estimate used in the feasibility page.")
    with b:
        fu = pd.DataFrame({'Stage': ['All respondents', 'Likelihood 3 or more', 'Likelihood 4 or 5', 'Joined pre-launch list'],
                           'Count': [len(F), (F.q24_likelihood >= 3).sum(), (F.q24_likelihood >= 4).sum(), F.q29_prelaunch_signup.sum()]})
        show(px.funnel(fu, x='Count', y='Stage', title="Purchase funnel"))
        note("How many people remain at each stage.", "A funnel shows drop-off between stages.", f"Only {fu.Count[3] / fu.Count[0]:.0%} of everyone joins the pre-launch list.", "Stated interest overstates real buying, so the feasibility page applies a conversion haircut.")
    at = F[ATT].mean().sort_values(); at.index = [i[4:].replace('_', ' ') for i in at.index]
    show(px.bar(at, orientation='h', range_x=[1, 5], title="Average agreement with each statement (1 to 5)", labels={'value': 'Mean score', 'index': ''}))
    note("Mean agreement with the attitude statements.", "Means summarize Likert items; compare them on one scale.", f"Highest: {at.idxmax()}; lowest: {at.idxmin()}.", "Build brand messaging around the highest-scoring attitudes.")

elif page == PAGES[3]:
    st.header("Correlation and diagnostic EDA: why do people differ?")
    if len(F) < 50: st.warning("Pick more traveler types: too few rows."); st.stop()
    cols = ['q5_income_band', 'q6_treks_per_year', 'q7_trip_length', 'q8_altitude', 'n_problems', 'q11_severity', 'q14_novelty', 'q15_natural',
            'q16_plastic_free', 'q17_himalayan_origin', 'q21_gifts', 'q22_price_priority', 'q24_likelihood', 'spend']
    C = F[cols].corr(method='spearman')
    show(px.imshow(C.round(2), text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title="Spearman correlation heatmap"))
    top = C.where(np.triu(np.ones(C.shape), 1).astype(bool)).stack().sort_values(key=abs, ascending=False).head(3)
    note("How strongly each pair of variables moves together (-1 to +1).", "Spearman suits 1-to-5 scales because it uses ranks, not exact distances.",
         "Strongest pairs: " + "; ".join(f"{a} and {b} ({v:.2f})" for (a, b), v in top.items()) + ".", "Variables that move together, such as problems and severity, overlap, so one of them is enough in a model. Correlation is not causation.")
    ct = pd.DataFrame([(c, *stats.spearmanr(F[c], F.q24_likelihood)) for c in cols if c not in ('q24_likelihood', 'spend')], columns=['Variable', 'Spearman rho', 'p-value']).sort_values('Spearman rho')
    show(px.bar(ct, x='Spearman rho', y='Variable', orientation='h', title="Correlation with likelihood to buy"))
    note("Which variables move with purchase intent.", "A sorted bar ranks drivers in one view.", f"Strongest positive: {ct.iloc[-1].Variable} ({ct.iloc[-1]['Spearman rho']:.2f}); strongest negative: {ct.iloc[0].Variable} ({ct.iloc[0]['Spearman rho']:.2f}).", "Target people who trek often and have more problems. Price-focused buyers are a weaker target.")
    st.dataframe(ct.round(4), width='stretch')
    a, b = st.columns(2)
    with a:
        show(px.box(F, x='q6_treks_per_year', y='q24_likelihood', title="Likelihood by treks per year (0 = none, 4 = 7+)"))
        kw = stats.kruskal(*[g.q24_likelihood for _, g in F.groupby('q6_treks_per_year')])
        note("Spread of purchase intent at each trekking frequency.", "Box plots compare a score across groups; Kruskal-Wallis tests whether the groups differ.", f"Kruskal-Wallis p = {kw.pvalue:.4f}: {'a significant' if kw.pvalue < .05 else 'no significant'} difference.", "Frequent trekkers are the core customers.")
    with b:
        rows = []
        for p in PROB:
            w, wo = F[F[p] == 1].interested_flag, F[F[p] == 0].interested_flag
            rows.append({'Problem': p[4:].replace('_', ' '), 'With problem': w.mean(), 'Without': wo.mean(), 'p-value': stats.chi2_contingency(pd.crosstab(F[p], F.interested_flag))[1]})
        pdf = pd.DataFrame(rows); show(px.bar(pdf.melt('Problem', ['With problem', 'Without'], value_name='Share interested'), x='Problem', y='Share interested', color='variable', barmode='group', title="Interest with vs without each problem"))
        note("Share interested, for people who do and do not face each problem.", "Grouped bars put the two groups side by side; a chi-square test checks each gap.", f"All gaps are {'significant' if (pdf['p-value'] < .05).all() else 'not all significant'} at 5%; the biggest is {pdf.assign(g=pdf['With problem'] - pdf.Without).sort_values('g').iloc[-1].Problem}.", "Prioritise products that fix the problems that raise interest most.")
    a, b = st.columns(2)
    with a:
        F2 = F.assign(price_band=pd.cut(F.q22_price_priority, [0, 2, 3, 5], labels=['Low', 'Neutral', 'High']))
        tab = pd.crosstab(F2.price_band, F2.interested_flag); chi = stats.chi2_contingency(tab); v = np.sqrt(chi[0] / (tab.values.sum() * (min(tab.shape) - 1)))
        show(px.bar(F2.groupby('price_band', observed=True).interested_flag.mean(), title="Share interested by price priority", labels={'value': 'Share interested', 'price_band': 'Price is top priority'}))
        note("Interest among low, neutral and high price-focus respondents.", "Chi-square tests whether the two categorical variables are linked; Cramer's V gives the strength.", f"Chi-square p = {chi[1]:.4f}, Cramer's V = {v:.2f}.", "Price-focused buyers are less interested, so lead with quality and sustainability, not discounts.")
    with b:
        m = F[F.spend > 0].copy(); m['Income'] = m.q5_income_band.map(lambda i: INC[CUR][i - 1])
        gg = m.groupby(['Income', 'q4_traveler_type']).spend.mean().reset_index()
        show(px.bar(gg, x='Income', y='spend', color='q4_traveler_type', barmode='group', category_orders={'Income': INC[CUR]}, title=f"Mean spend by income band and traveler type ({CUR})"))
        note("Average spend across income bands, split by traveler type.", "Grouped bars check whether country or income explains low spend (a confounder check).", "Within the same income band, Nepal residents still spend less than other groups, so market and income both matter.", "Price the line differently for Nepal, for example smaller sizes, rather than assuming one price fits all.")
    mynotes("n_diag")

elif page == PAGES[4]:
    st.header("Classification: will a customer be interested?")
    why_box("**Why classification:** the outcome is yes or no (interested or not), so a classifier can score each person and find the traits that predict interest. "
            "**Why 8 algorithms:** each makes different assumptions, so comparing them shows which fits this data. "
            "**Why tuning:** default settings are rarely best. GridSearchCV tries every combination with 5-fold cross-validation on the training data, and the untouched test set is used once at the end.")
    r = run_clf(); t = r['table']
    st.caption(f"Training rows: {r['ntr']:,}; test rows: {r['nte']:,}. Target: interested_flag (likelihood 4 or 5). Predictors exclude all outcome answers to avoid leakage.")
    st.dataframe(t.style.format({c: '{:.3f}' for c in t.columns if c not in ('Model', 'Best parameters')}), width='stretch')
    show(px.bar(t.melt('Model', ['Test accuracy', 'Precision', 'Recall', 'F1', 'AUC'], var_name='Metric'), x='Model', y='value', color='Metric', barmode='group', title="Tuned model comparison on the test set"))
    note("Test-set accuracy, precision, recall, F1 and AUC for each tuned model.", "Grouped bars compare all metrics at once.", f"Best accuracy: {t.Model[0]} ({t['Test accuracy'][0]:.1%}). Tuning mattered most for models sensitive to settings, such as KNN and Decision Tree.", "Use the best model's scores to estimate how many prospects are likely to convert.")
    a, b = st.columns(2)
    with a:
        fig = go.Figure([go.Scatter(x=f, y=tp, name=n, mode='lines') for n, (f, tp) in r['roc'].items()] + [go.Scatter(x=[0, 1], y=[0, 1], name='Chance', line=dict(dash='dash'))])
        fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate"); show(fig)
    with b:
        mdl = st.selectbox("Confusion matrix for", t.Model.tolist()); c = r['cm'][mdl]
        show(px.imshow(c, text_auto=True, x=['Predicted no', 'Predicted yes'], y=['Actual no', 'Actual yes'], color_continuous_scale='Greens', title=f"Confusion matrix: {mdl}"))
    st.write("**Reading the metrics:** accuracy = share of correct predictions. Precision = of those predicted interested, how many really were. Recall = of those really interested, how many we found. F1 balances precision and recall. AUC = how well the model ranks interested people above others (0.5 is chance, 1.0 is perfect).")
    mynotes("n_clf")

elif page == PAGES[5]:
    st.header("Clustering: customer segments")
    why_box("**Why clustering:** we do not tell the algorithm who is interested. K-Means groups similar people using behaviour and attitudes only, then we check how interested each group is. "
            "**Tuning:** the number of clusters k, the initialisation method and n_init are tested, scored by silhouette (higher means better separated groups).")
    cs = clu_scores(); show(px.line(cs, x='k', y='silhouette', color='init', markers=True, title="Silhouette score by k"))
    show(px.line(cs[cs.init == 'k-means++'], x='k', y='inertia', markers=True, title="Elbow curve (inertia)"))
    best = cs.sort_values('silhouette').iloc[-1]
    st.write(f"Highest silhouette is at k = {int(best.k)}. Scores are low overall (under 0.2), meaning the groups overlap, which is normal for survey data. We use **k = 3** (Less, Medium, Most probable) because it gives a clear business gradient.")
    k = st.slider("Number of clusters", 2, 6, 3); lab, pc = clu_labels(k)
    prof = D.assign(Segment=lab).groupby('Segment').agg(Share=('respondent_id', lambda s: len(s) / len(D)), Interested=('interested_flag', 'mean'), Avg_spend=('spend', 'mean'), Treks=('q6_treks_per_year', 'mean'),
                                                       Problems=('n_problems', 'mean'), Natural=('q15_natural', 'mean'), Price_priority=('q22_price_priority', 'mean')).sort_values('Interested')
    st.dataframe(prof.style.format({'Share': '{:.0%}', 'Interested': '{:.0%}', 'Avg_spend': '{:,.0f}', 'Treks': '{:.2f}', 'Problems': '{:.2f}', 'Natural': '{:.2f}', 'Price_priority': '{:.2f}'}), width='stretch')
    smp = np.random.default_rng(1).choice(len(D), 1200, replace=False)
    show(px.scatter(x=pc[smp, 0], y=pc[smp, 1], color=lab.values[smp], labels={'x': 'PC1', 'y': 'PC2', 'color': 'Segment'}, title="Segments projected to 2D (PCA, 1,200 sampled respondents)"))
    note("Segments built only from behaviour and attitude answers, then profiled.", "A table shows what defines each group; PCA squeezes 13 features into 2 axes so the groups can be seen.", "Interest rises steeply from the Less to the Most probable segment, and the Most probable group trek more, face more problems and care more about natural products.", "Spend marketing money on the Most probable segment first; use education and sampling for the Medium group.")
    mynotes("n_clu")

elif page == PAGES[6]:
    st.header("Association rules: which products are wanted together?")
    why_box("**Why association rules:** the survey asked about 12 products. Rules such as 'if someone wants A and B, they also want C' show which products to bundle. "
            "**Measures:** support = how common the combination is; confidence = how often C follows A and B; lift = how much more than chance (above 1 means a real link). "
            "**Tuning:** minimum support, confidence and lift decide how many rules survive. Too low gives noise, too high loses insight, so a grid of settings is tested.")
    S = supports()
    a, b, c = st.columns(3); ms = a.slider("Minimum support", .05, .30, .10, .01); mc = b.slider("Minimum confidence", .3, .9, .5, .05); ml = c.slider("Minimum lift", 1.0, 2.0, 1.1, .05)
    R = rules(S, ms, mc, ml).sort_values('Lift', ascending=False)
    st.write(f"**{len(R)} rules** pass these settings. Top 15 by lift:")
    st.dataframe(R.head(15).style.format({'Support': '{:.1%}', 'Confidence': '{:.1%}', 'Lift': '{:.2f}'}), width='stretch')
    gr = rule_grid(); show(px.imshow(gr.pivot(index='min support', columns='min confidence', values='rules found'), text_auto=True, color_continuous_scale='Blues', title="Rules found for each setting (tuning grid)"))
    nb = D0.n_products_wanted; ok = nb[nb >= 1]
    show(px.histogram(nb, nbins=13, title="How many of the 12 products people say they would buy"))
    note("The strongest product links, and how many products a person typically wants.", "A rule table ranks links; the histogram shows basket size.", f"Among people who want any product, {(ok >= 2).mean():.0%} want 2 or more and {(ok >= 3).mean():.0%} want 3 or more; the average is {ok.mean():.1f}.", "Bundle the products with the highest lift, and offer the mini kit as a cross-sell.")
    mynotes("n_assoc")

elif page == PAGES[7]:
    st.header("Regression: how much will customers spend?")
    why_box("**Why regression:** spend is a number, so regression predicts how much a likely buyer will spend and which factors push it up or down. "
            "**Why Linear, Ridge and Lasso:** Linear is the baseline; Ridge shrinks coefficients to cope with overlapping predictors; Lasso shrinks some to exactly zero, so it also selects features. "
            "**Tuning:** the penalty strength alpha is chosen by 5-fold cross-validation. Spend is modelled on a log scale because it is right-skewed.")
    r = run_reg(); t = r['table']; st.caption(f"{r['nb']:,} likely buyers (spend above zero). Predictors include stated likelihood to buy.")
    st.dataframe(t.style.format({c: '{:,.3f}' for c in t.columns if 'R2' in c} | {'Best alpha': '{:.4f}', 'MAE (NPR)': '{:,.0f}', 'RMSE (NPR)': '{:,.0f}'}), width='stretch')
    cf = r['coef'][r['coef'] != 0].sort_values(); show(px.bar(cf.tail(12), orientation='h', title="Strongest Lasso coefficients (standardised, log-spend)"))
    show(px.scatter(r['avp'], x='Actual', y='Predicted', title="Actual vs predicted spend (NPR, test set)", opacity=.5))
    note("Which factors raise or lower spend, and how well the model predicts it.", "Coefficients show direction and size; the scatter shows fit (points on the diagonal are perfect).", f"{r['best']} fits best. R2 on the log scale is {t['Test R2 (log)'].max():.2f}; on raw NPR it is lower ({t['Test R2 (NPR)'].max():.2f}) because big spenders are harder to predict. Lasso dropped {int(t.loc[t.Model == 'Lasso', 'Features dropped'].iloc[0])} weak features.",
         "Use predicted spend by segment, not one average, when forecasting revenue.")
    mynotes("n_reg")

else:
    st.header("Feasibility verdict: is the idea sustainable?")
    st.write("All four algorithms feed this page: **classification** gives each segment's chance of being interested, **regression** gives expected spend, **clustering** shows where demand is concentrated, and **association rules** say what to bundle.")
    clf, reg = run_clf(), run_reg()
    G = D.assign(p=clf['pall'], sp=reg['spend_all'] * FX, seg=clu_labels(3)[0])
    T = pd.DataFrame({'p': G.groupby('q4_traveler_type').p.mean(), 'spend': G[G.p >= .5].groupby('q4_traveler_type').sp.mean()}).fillna(0)
    st.dataframe(T.rename(columns={'p': 'Predicted interest (classification)', 'spend': f'Predicted spend per buyer, {CUR} (regression)'}).style.format({'Predicted interest (classification)': '{:.0%}', f'Predicted spend per buyer, {CUR} (regression)': '{:,.0f}'}))
    sg = G.groupby('seg').agg(Share=('p', lambda s: len(s) / len(G)), Interest=('p', 'mean')).sort_values('Interest')
    st.write("**Clustering:** " + "; ".join(f"{i} = {r.Share:.0%} of people, {r.Interest:.0%} predicted interest" for i, r in sg.iterrows()))
    R = rules(supports(), .1, .5, 1.1).sort_values('Lift', ascending=False)
    if len(R): st.write(f"**Association rules:** launch the bundle '{R.iloc[0]['If customer wants']} + {R.iloc[0]['Then also wants']}' (lift {R.iloc[0].Lift:.2f}).")
    st.subheader("Business assumptions (change them)")
    u = 1 / FX; rd = lambda v: int(round(v / u, -3)) if CUR == 'AED' else int(round(v, -4))
    a, b, c = st.columns(3)
    conv = a.slider("Stated interest that becomes a purchase", .05, .5, .2, .01); reach = b.slider("Share of reachable market served by year 3", .05, .6, .25, .01); real = c.slider("Real spend vs stated spend", .5, 1.0, .85, .01)
    a, b, c = st.columns(3)
    gm = a.slider("Gross margin", .3, .75, .55, .01); fix = b.number_input(f"Fixed costs per year ({CUR})", value=rd(6_000_000), step=rd(500_000)); cap = c.number_input(f"Setup investment ({CUR})", value=rd(8_000_000), step=rd(500_000))
    st.caption("Defaults: fixed costs NPR 6 million and setup NPR 80 lakh per year (placeholders, replace with real quotes). Reachable-market sizes below are placeholders too.")
    pool = {t: st.number_input(f"Reachable customers per year: {t}", value=v, step=1000) for t, v in {'International trekker': 80000, 'Nepal resident': 60000, 'UAE resident': 40000, 'Nepali diaspora': 25000}.items()}
    def fin(conv, reach, real, gm):
        out, cum = [], -cap
        for rp in (.3, .6, 1):
            cu = sum(pool.get(t, 0) * r.p * conv * reach * rp for t, r in T.iterrows()); rev = sum(pool.get(t, 0) * r.p * conv * reach * rp * r.spend * real for t, r in T.iterrows())
            cum += rev * gm - fix; out.append(dict(cust=cu, rev=rev, cost=rev * (1 - gm) + fix, profit=rev * gm - fix, cum=cum))
        return out
    ys = fin(conv, reach, real, gm); y3 = ys[2]; be = fix / ((y3['rev'] / y3['cust']) * gm) if y3['cust'] else 0
    if y3['profit'] > 0 and y3['cum'] >= 0: st.success("**Viable on these assumptions:** profitable in year 3 and the setup cost is paid back within 3 years.")
    elif y3['profit'] > 0: st.warning("**Conditional:** profitable in year 3, but payback takes longer than 3 years.")
    else: st.error("**Not sustainable on these assumptions:** year 3 does not cover fixed costs.")
    c = st.columns(4); c[0].metric("Year-3 customers", f"{y3['cust']:,.0f}"); c[1].metric("Year-3 revenue", money(y3['rev'])); c[2].metric("Year-3 net profit", money(y3['profit'])); c[3].metric("Break-even customers per year", f"{be:,.0f}")
    fig = go.Figure([go.Bar(x=['Year 1', 'Year 2', 'Year 3'], y=[y['rev'] for y in ys], name='Revenue'), go.Bar(x=['Year 1', 'Year 2', 'Year 3'], y=[y['cost'] for y in ys], name='Total costs'),
                     go.Scatter(x=['Year 1', 'Year 2', 'Year 3'], y=[y['cum'] for y in ys], name='Cumulative cash (after setup)', mode='lines+markers')])
    fig.update_layout(title=f"Three-year outlook ({CUR})"); show(fig)
    cv, rc = [.1, .15, .2, .25, .3], [.1, .15, .25, .35, .45]
    grid = pd.DataFrame([[fin(c_, r_, real, gm)[2]['profit'] for r_ in rc] for c_ in cv], index=[f"{x:.0%}" for x in cv], columns=[f"{x:.0%}" for x in rc])
    show(px.imshow(grid.round(0), text_auto=',.0f', color_continuous_scale='RdYlGn', color_continuous_midpoint=0, aspect='auto', labels=dict(x='Year-3 reach', y='Conversion', color='Net profit'), title=f"Year-3 net profit ({CUR}) by conversion and reach"))
    note("Three-year revenue, costs and cash, and how profit changes with conversion and reach.", "A combined bar and line chart shows both annual results and payback; the heatmap shows which assumptions matter most.", "The verdict flips between profit and loss across the grid, so success depends on how many prospects become buyers and how much of the market is reached.", "Validate conversion and reach with real pre-launch data before buying machinery.")
    st.caption("Limitation: the survey is synthetic, so these results show how the method works, not real demand. Re-run the app on real responses before committing capital.")
    mynotes("n_feas")
