"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       SCOUT VISION — I-League Player Selection ML System                    ║
║       Dataset: i_league_players_dataset.xlsx                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

PIPELINE:
  1.  Data Loading & Preprocessing        (openpyxl + pandas)
  2.  Automated Performance Report        (descriptive stats + violin plots)
  3.  Player Similarity Analysis          (K-Nearest Neighbours)
  4.  Scout Selection Model               (GradientBoosting ≡ XGBoost, RF, LR)
  5.  Match Analysis                      (club-level attack/defense/mid/gk scores)
  6.  Match Outcome Prediction            (GradientBoosting classifier)
  7.  Tactical Recommendations            (rule-based on score profiles)
  8.  Radar Charts                        (top players per position)
  9.  Export                              (CSV reports + saved .pkl models)

Algorithm notes:
  • GradientBoostingClassifier  → scikit-learn's XGBoost equivalent; sequential
    boosted trees are ideal for small tabular sports datasets with mixed scales.
  • RandomForestClassifier       → ensemble baseline; robust to outliers.
  • KNearestNeighbors            → distance-based similarity in metric space;
    used by scouts to find replacement / comparable player profiles.
  • PCA                          → 2-D projection for visual similarity maps.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import warnings
import itertools
from collections import defaultdict
from datetime import datetime

import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
import matplotlib.patches   as mpatches
import seaborn              as sns

from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.neighbors        import NearestNeighbors
from sklearn.decomposition    import PCA
from sklearn.ensemble         import (GradientBoostingClassifier,
                                      RandomForestClassifier)
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import cross_val_score, StratifiedKFold
from sklearn.metrics          import classification_report, confusion_matrix
from sklearn.pipeline         import Pipeline
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE = "C:\\Users\\sheri\\OneDrive\\Desktop\\project final year\\champions_league_players_dataset.xlsx"
OUT_DIR   = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

# Colour palette
C_BG     = '#0d1117'
C_CARD   = '#161b24'
C_BORDER = '#2a3040'
C_TEXT   = '#e8edf5'
C_MUTED  = '#8a95a8'
C_SELECT = '#00e5ff'
C_WATCH  = '#ffd700'
C_PASS   = '#5a6478'

POS_COLOR = {
    'Goalkeeper': '#ff6b35',
    'Defender':   '#00c8ff',
    'Midfielder': '#7cfc00',
    'Attacker':   '#ffd700',
}

VERDICT_COLOR = {'SELECT': C_SELECT, 'WATCH': C_WATCH, 'PASS': C_PASS}

# ── Position metric definitions & scouting standards ─────────────────────────
#   Standards are set at ~70th percentile of the dataset so roughly the
#   top 30 % of players will be flagged SELECT — a realistic scout shortlist.
POS_CONFIG = {
    'Goalkeeper': {
        'metrics':  ['xSaves_per90', 'PostShot_xG', 'ShotsFaced_per90'],
        'labels':   ['xSaves/90',    'PostShot xG',  'Shots Faced/90'],
        'standards': {
            'xSaves_per90':      3.50,   # top ~30 % threshold
            'PostShot_xG':       0.44,
            'ShotsFaced_per90':  5.20,   # higher = busier/tougher test
        },
        'higher_better': {
            'xSaves_per90':     True,
            'PostShot_xG':      True,
            'ShotsFaced_per90': True,   # more shots faced → tougher test passed
        },
        'weights': {
            'xSaves_per90':     0.50,
            'PostShot_xG':      0.30,
            'ShotsFaced_per90': 0.20,
        },
    },
    'Defender': {
        'metrics': ['Tackles_per90', 'TackleSuccessRate', 'Interceptions',
                    'Clearances', 'AerialDuelsWon', 'Recoveries'],
        'labels':  ['Tackles/90', 'Tackle %', 'Interceptions',
                    'Clearances', 'Aerial/90', 'Recoveries'],
        'standards': {
            'Tackles_per90':    2.30,
            'TackleSuccessRate':70.0,
            'Interceptions':     2.00,
            'Clearances':        5.50,
            'AerialDuelsWon':    3.20,
            'Recoveries':        7.20,
        },
        'higher_better': {m: True for m in
                          ['Tackles_per90','TackleSuccessRate','Interceptions',
                           'Clearances','AerialDuelsWon','Recoveries']},
        'weights': {
            'Tackles_per90':    0.20,
            'TackleSuccessRate':0.25,
            'Interceptions':    0.18,
            'Clearances':       0.15,
            'AerialDuelsWon':   0.12,
            'Recoveries':       0.10,
        },
    },
    'Midfielder': {
        'metrics': ['KeyPasses_per90', 'ProgressiveCarries',
                    'Touches_in_PenaltyArea', 'NonPenaltyGoals_per90'],
        'labels':  ['Key Passes/90', 'Prog. Carries',
                    'Box Touches',   'NP Goals/90'],
        'standards': {
            'KeyPasses_per90':        2.20,
            'ProgressiveCarries':     7.00,
            'Touches_in_PenaltyArea': 5.00,
            'NonPenaltyGoals_per90':  0.14,
        },
        'higher_better': {m: True for m in
                          ['KeyPasses_per90','ProgressiveCarries',
                           'Touches_in_PenaltyArea','NonPenaltyGoals_per90']},
        'weights': {
            'KeyPasses_per90':        0.35,
            'ProgressiveCarries':     0.28,
            'Touches_in_PenaltyArea': 0.22,
            'NonPenaltyGoals_per90':  0.15,
        },
    },
    'Attacker': {
        'metrics': ['Goals_per90', 'Assists_per90'],
        'labels':  ['Goals/90',    'Assists/90'],
        'standards': {
            'Goals_per90':   0.46,
            'Assists_per90': 0.20,
        },
        'higher_better': {'Goals_per90': True, 'Assists_per90': True},
        'weights':        {'Goals_per90': 0.65, 'Assists_per90': 0.35},
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_xlsx(path: str) -> pd.DataFrame:
    """
    Read the Excel file using openpyxl and return a clean DataFrame.
    openpyxl is used directly so we can handle .xlsx without xlrd dependency.
    """
    try:
        import openpyxl
    except ImportError:
        sys.exit("❌  openpyxl not found. Run:  pip install openpyxl")

    wb  = openpyxl.load_workbook(path)
    ws  = wb.active
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) for h in rows[0]]
    df  = pd.DataFrame(rows[1:], columns=headers)

    # Coerce numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    print(f"✔  Loaded '{path}'  →  {len(df)} players,  "
          f"{df['Position'].nunique()} positions")
    return df


def assign_label(row: pd.Series, pos: str) -> str:
    """
    Rule-based scouting label used as supervised ground truth.
      SELECT  → meets ALL position standards           (scout's first choice)
      WATCH   → meets ≥ 60 % of standards              (worth monitoring)
      PASS    → below threshold                         (not ready)
    Rationale: encoding expert scout judgment as labels lets the ML model
    learn to generalise the decision boundary to unseen players.
    """
    cfg = POS_CONFIG[pos]
    met = sum(
        1 for m in cfg['metrics']
        if pd.notna(row.get(m)) and (
            row[m] >= cfg['standards'][m] if cfg['higher_better'][m]
            else row[m] <= cfg['standards'][m]
        )
    )
    ratio = met / len(cfg['metrics'])
    return 'SELECT' if ratio == 1.0 else 'WATCH' if ratio >= 0.6 else 'PASS'


def composite_score(sub: pd.DataFrame, pos: str) -> pd.Series:
    """Weighted normalised composite — higher always means better scout value."""
    cfg   = POS_CONFIG[pos]
    score = pd.Series(0.0, index=sub.index)
    for m in cfg['metrics']:
        mn, mx = sub[m].min(), sub[m].max()
        norm   = (sub[m] - mn) / (mx - mn + 1e-9)
        score += (norm if cfg['higher_better'][m] else 1 - norm) * cfg['weights'][m]
    return score


def preprocess(df: pd.DataFrame) -> dict:
    """Split by position, drop incomplete rows, assign labels & composite scores."""
    print("\n" + "═"*62)
    print("  MODULE 1 — DATA PREPROCESSING")
    print("═"*62)
    pos_data = {}
    for pos, cfg in POS_CONFIG.items():
        sub = df[df['Position'] == pos].copy().dropna(subset=cfg['metrics'])
        sub = sub.reset_index(drop=True)
        if sub.empty:
            print(f"  ⚠  No complete rows for {pos}"); continue
        sub['Label']           = sub.apply(lambda r: assign_label(r, pos), axis=1)
        sub['Composite_Score'] = composite_score(sub, pos)
        pos_data[pos] = sub
        vc = sub['Label'].value_counts()
        print(f"  {pos:<12}  {len(sub):>3} players │ "
              f"SELECT={vc.get('SELECT',0)}  "
              f"WATCH={vc.get('WATCH',0)}  "
              f"PASS={vc.get('PASS',0)}")
    return pos_data


def engineer_features(sub: pd.DataFrame, pos: str) -> pd.DataFrame:
    """
    Create derived features on top of raw metrics:
      - raw metrics themselves
      - signed distance from standard  (positive = above threshold)
      - direction-corrected normalised value  (0–1, higher always better)
      - overall composite score
    These features give ensemble tree models cleaner split points and help
    logistic regression with scaling.
    """
    cfg  = POS_CONFIG[pos]
    mets = cfg['metrics']
    X    = sub[mets].copy().astype(float)

    for m in mets:
        diff       = X[m] - cfg['standards'][m]
        X[f'{m}_vs_std'] = diff if cfg['higher_better'][m] else -diff

        mn, mx = X[m].min(), X[m].max()
        norm   = (X[m] - mn) / (mx - mn + 1e-9)
        X[f'{m}_norm'] = norm if cfg['higher_better'][m] else 1 - norm

    X['composite'] = sum(X[f'{m}_norm'] * cfg['weights'][m] for m in mets)
    return X


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — AUTOMATED PERFORMANCE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def performance_report(pos_data: dict) -> pd.DataFrame:
    """Descriptive statistics + distribution visualisations per position."""
    print("\n" + "═"*62)
    print("  MODULE 2 — AUTOMATED PERFORMANCE REPORT")
    print("═"*62)

    all_desc = []
    for pos, sub in pos_data.items():
        mets = POS_CONFIG[pos]['metrics']
        desc = sub[mets].describe().T.round(4)
        desc.insert(0, 'Position', pos)
        all_desc.append(desc)
        print(f"\n  [{pos}]")
        print(desc[['mean','std','min','50%','max']].to_string())

    summary = pd.concat(all_desc)
    summary.to_csv(os.path.join(OUT_DIR, 'descriptive_stats.csv'))

    _plot_distributions(pos_data)
    return summary


def _plot_distributions(pos_data: dict):
    n_pos  = len(pos_data)
    fig    = plt.figure(figsize=(24, 5 * n_pos), facecolor=C_BG)
    fig.suptitle('I-League — Performance Distributions by Position & Metric',
                 color=C_TEXT, fontsize=15, fontweight='bold', y=1.005)
    outer  = gridspec.GridSpec(n_pos, 1, figure=fig, hspace=0.6)

    for row_i, (pos, sub) in enumerate(pos_data.items()):
        cfg   = POS_CONFIG[pos]
        mets  = cfg['metrics']
        color = POS_COLOR[pos]
        inner = gridspec.GridSpecFromSubplotSpec(
            1, len(mets), subplot_spec=outer[row_i], wspace=0.35)

        for col_j, (m, lbl) in enumerate(zip(mets, cfg['labels'])):
            ax = fig.add_subplot(inner[col_j])
            ax.set_facecolor(C_CARD)
            for sp in ax.spines.values(): sp.set_edgecolor(C_BORDER)
            ax.tick_params(colors=C_MUTED, labelsize=7)

            data = sub[m].dropna().values
            parts = ax.violinplot(data, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(color); pc.set_alpha(0.35)
            parts['cmedians'].set_color(color); parts['cmedians'].set_linewidth(2)
            for k in ('cbars','cmins','cmaxes'): parts[k].set_color(C_BORDER)

            jitter = np.random.uniform(-0.09, 0.09, len(sub))
            dot_c  = sub['Label'].map(VERDICT_COLOR)
            ax.scatter(1 + jitter, sub[m], c=dot_c, s=22, alpha=0.85, zorder=4)

            std_val = cfg['standards'][m]
            ax.axhline(std_val, color='white', ls='--', lw=0.9, alpha=0.55)
            ax.text(1.42, std_val, f'{std_val}', color='white',
                    fontsize=6.5, va='center', alpha=0.75)

            ax.set_title(lbl, color=color, fontsize=9, pad=5)
            ax.set_xticks([])
            if col_j == 0:
                ax.set_ylabel(pos, color=color, fontsize=10, fontweight='bold')

    legend_patches = [mpatches.Patch(color=VERDICT_COLOR[v], label=v)
                      for v in ('SELECT','WATCH','PASS')]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=9)

    path = os.path.join(OUT_DIR, 'report_distributions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print(f"\n  📊 Distribution plot  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — PLAYER SIMILARITY  (K-Nearest Neighbours)
# ══════════════════════════════════════════════════════════════════════════════
# WHY KNN? A scout's core workflow is "find me players like X."
# KNN in standardised metric-space gives the closest statistical neighbours
# — ideal for transfer recommendations and unknown player benchmarking.

def player_similarity(pos_data: dict, k: int = 5) -> dict:
    """Fit KNN per position; export similarity tables + PCA maps."""
    print("\n" + "═"*62)
    print("  MODULE 3 — PLAYER SIMILARITY (KNN + PCA visualisation)")
    print("═"*62)

    results = {}
    for pos, sub in pos_data.items():
        mets   = POS_CONFIG[pos]['metrics']
        scaler = StandardScaler()           # mandatory: KNN uses Euclidean distance
        X_sc   = scaler.fit_transform(sub[mets].astype(float))

        n_n = min(k + 1, len(sub))
        knn = NearestNeighbors(n_neighbors=n_n, metric='euclidean', algorithm='auto')
        knn.fit(X_sc)
        dists, idxs = knn.kneighbors(X_sc)

        rows = []
        for i in range(len(sub)):
            top3_names = [sub.iloc[j]['Name'] for j in idxs[i][1:4]]
            top3_dists = [round(d, 3)          for d in dists[i][1:4]]
            rows.append({
                'Rank':            i + 1,
                'Player':          sub.iloc[i]['Name'],
                'Club':            sub.iloc[i]['Club'],
                'Verdict':         sub.iloc[i]['Label'],
                'Score':           round(sub.iloc[i]['Composite_Score'], 4),
                'Similar_Players': ' | '.join(top3_names),
                'Distances':       str(top3_dists),
            })

        df_sim = pd.DataFrame(rows)
        results[pos] = df_sim
        df_sim.to_csv(os.path.join(OUT_DIR, f'similarity_{pos.lower()}.csv'), index=False)

        print(f"\n  [{pos}]  Sample similarities:")
        print(df_sim[['Player','Similar_Players']].head(6).to_string(index=False))

        joblib.dump({'knn': knn, 'scaler': scaler, 'names': sub['Name'].tolist()},
                    os.path.join(OUT_DIR, f'knn_{pos.lower()}.pkl'))
        _plot_similarity_pca(sub, X_sc, pos)

    return results


def _plot_similarity_pca(sub: pd.DataFrame, X_sc: np.ndarray, pos: str):
    pca  = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X_sc)
    ev   = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=C_BG)
    ax.set_facecolor(C_CARD)
    for sp in ax.spines.values(): sp.set_edgecolor(C_BORDER)
    ax.tick_params(colors=C_MUTED)

    for verdict, grp in sub.groupby('Label'):
        idx = grp.index
        ax.scatter(proj[idx, 0], proj[idx, 1],
                   c=VERDICT_COLOR[verdict], label=verdict,
                   s=85, alpha=0.88, edgecolors='white', linewidths=0.4)

    # annotate SELECT players
    for i in sub[sub['Label'] == 'SELECT'].index:
        ax.annotate(sub.loc[i, 'Name'],
                    (proj[i, 0], proj[i, 1]),
                    fontsize=6.5, color='white', alpha=0.85,
                    xytext=(5, 5), textcoords='offset points')

    ax.set_title(f'{pos}  —  Player Similarity Map  (PCA projection)',
                 color=POS_COLOR[pos], fontsize=13, fontweight='bold')
    ax.set_xlabel(f'PC1  ({ev[0]*100:.1f} % variance)', color=C_MUTED)
    ax.set_ylabel(f'PC2  ({ev[1]*100:.1f} % variance)', color=C_MUTED)
    ax.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=9)

    path = os.path.join(OUT_DIR, f'similarity_{pos.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print(f"  📊 PCA map  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — SCOUT SELECTION MODEL
# ══════════════════════════════════════════════════════════════════════════════
# WHY GradientBoosting?  It sequentially corrects residual errors of shallow
# trees — the same core algorithm as XGBoost — and excels on small, tabular
# datasets with non-linear metric interactions.
# Random Forest & Logistic Regression compete as baselines; best F1 wins.

def train_selection_models(pos_data: dict) -> dict:
    """Train per-position scout-selection classifiers. Save winners to disk."""
    print("\n" + "═"*62)
    print("  MODULE 4 — SCOUT SELECTION MODELS")
    print("═"*62)

    store   = {}
    cv_rows = []

    for pos, sub in pos_data.items():
        print(f"\n  ── {pos} ──")
        X   = engineer_features(sub, pos)
        le  = LabelEncoder()
        y   = le.fit_transform(sub['Label'])

        # Three competing classifiers
        candidates = {
            'GradientBoosting (≡ XGBoost)': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08,
                max_depth=3, subsample=0.85, random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=300, class_weight='balanced',
                max_features='sqrt', random_state=42),
            'Logistic Regression': Pipeline([
                ('sc', StandardScaler()),
                ('lr', LogisticRegression(C=1.0, class_weight='balanced',
                                          max_iter=1000, random_state=42)),
            ]),
        }

        min_cls  = int(np.bincount(y).min())
        n_splits = max(2, min(5, min_cls))
        cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        print(f"  {n_splits}-fold stratified CV  ({len(sub)} players, "
              f"{len(le.classes_)} classes: {list(le.classes_)})")

        best_name, best_f1, best_model = None, -1, None
        for name, mdl in candidates.items():
            f1s  = cross_val_score(mdl, X, y, cv=cv, scoring='f1_weighted')
            mean = f1s.mean()
            print(f"    {name:<35}  F1={mean:.3f}  (±{f1s.std():.3f})")
            cv_rows.append({'Position': pos, 'Model': name,
                            'F1_mean': round(mean, 4), 'F1_std': round(f1s.std(), 4)})
            if mean > best_f1:
                best_f1, best_name, best_model = mean, name, mdl

        print(f"  ✅ Best → {best_name}  (F1={best_f1:.3f})")
        best_model.fit(X, y)
        y_pred = best_model.predict(X)

        print(f"\n  Classification Report:")
        print(classification_report(y, y_pred,
                                    target_names=le.classes_, zero_division=0))

        store[pos] = {'model': best_model, 'le': le,
                      'model_name': best_name, 'feature_cols': list(X.columns)}
        joblib.dump(store[pos], os.path.join(OUT_DIR, f'model_{pos.lower()}.pkl'))

        _plot_model_dashboard(pos, sub, X, y, y_pred, le, best_model, best_name)

    pd.DataFrame(cv_rows).to_csv(
        os.path.join(OUT_DIR, 'cv_results.csv'), index=False)
    return store


def _plot_model_dashboard(pos, sub, X, y, y_pred, le, model, model_name):
    fig = plt.figure(figsize=(20, 11), facecolor=C_BG)
    fig.suptitle(f'Scout Selection Model  —  {pos}  │  {model_name}',
                 color=C_TEXT, fontsize=14, fontweight='bold', y=0.99)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
    col = POS_COLOR[pos]

    def sax(ax, title=''):
        ax.set_facecolor(C_CARD)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(C_BORDER)
        ax.xaxis.label.set_color(C_MUTED)
        ax.yaxis.label.set_color(C_MUTED)
        if title: ax.set_title(title, color=C_TEXT, fontsize=10, pad=8)

    # ── A) Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sax(ax1, 'Confusion Matrix')
    cm  = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax1,
                xticklabels=le.classes_, yticklabels=le.classes_,
                cmap='YlOrRd', linewidths=0.5, linecolor=C_BG,
                annot_kws={'color': 'white', 'fontsize': 12})
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')
    ax1.tick_params(colors=C_MUTED)

    # ── B) Feature importance
    ax2 = fig.add_subplot(gs[0, 1])
    sax(ax2, 'Feature Importance')
    try:
        imp = (model.feature_importances_
               if hasattr(model, 'feature_importances_')
               else (model.named_steps['lr'].coef_[0]
                     if hasattr(model, 'named_steps') else model.coef_[0]))
        fd = (pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(imp)})
              .sort_values('Importance').tail(14))
        ax2.barh(fd['Feature'], fd['Importance'],
                 color=col, alpha=0.82, edgecolor=C_BG)
        ax2.set_xlabel('Importance')
    except Exception:
        ax2.text(0.5, 0.5, 'N/A', ha='center', va='center', color=C_MUTED, fontsize=13)

    # ── C) Composite score histogram by verdict
    ax3 = fig.add_subplot(gs[0, 2])
    sax(ax3, 'Composite Score Distribution')
    for lbl, grp in sub.groupby('Label'):
        ax3.hist(grp['Composite_Score'], bins=9, alpha=0.65,
                 label=lbl, color=VERDICT_COLOR[lbl], edgecolor=C_BG)
    ax3.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=8)
    ax3.set_xlabel('Composite Score')

    # ── D) Top 15 players bar
    ax4 = fig.add_subplot(gs[1, :2])
    sax(ax4, 'Top 15 Players  —  Composite Score')
    top15 = sub.nlargest(15, 'Composite_Score')
    bar_c = [VERDICT_COLOR[l] for l in top15['Label']]
    bars  = ax4.barh(top15['Name'], top15['Composite_Score'],
                     color=bar_c, alpha=0.86, edgecolor=C_BG)
    for bar, lbl in zip(bars, top15['Label']):
        ax4.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height()/2,
                 lbl, va='center', color=VERDICT_COLOR[lbl], fontsize=7.5)
    ax4.set_xlabel('Composite Score')

    # ── E) Scatter: top-2 metrics
    cfg  = POS_CONFIG[pos]
    mets = cfg['metrics']
    m1   = mets[0]; m2 = mets[min(1, len(mets)-1)]
    ax5  = fig.add_subplot(gs[1, 2])
    sax(ax5, f'{cfg["labels"][0]}  vs  {cfg["labels"][min(1,len(mets)-1)]}')
    for lbl, grp in sub.groupby('Label'):
        ax5.scatter(grp[m1], grp[m2], c=VERDICT_COLOR[lbl], label=lbl,
                    s=58, alpha=0.85, edgecolors='white', linewidths=0.35)
    ax5.axvline(cfg['standards'][m1], color='white', ls='--', lw=0.8, alpha=0.5)
    if len(mets) > 1:
        ax5.axhline(cfg['standards'][m2], color='white', ls='--', lw=0.8, alpha=0.5)
    ax5.set_xlabel(cfg['labels'][0])
    ax5.set_ylabel(cfg['labels'][min(1, len(mets)-1)])
    ax5.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=8)

    path = os.path.join(OUT_DIR, f'model_{pos.lower()}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print(f"  📊 Model dashboard  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — MATCH ANALYSIS  (club-level scores)
# ══════════════════════════════════════════════════════════════════════════════

def match_analysis(pos_data: dict) -> pd.DataFrame:
    """Aggregate per-club attacking, defensive, midfield, GK performance scores."""
    print("\n" + "═"*62)
    print("  MODULE 5 — MATCH ANALYSIS  (Club-Level Scores)")
    print("═"*62)

    club_scores = {}

    if 'Attacker' in pos_data:
        sub = pos_data['Attacker']
        club_scores['Attacking'] = sub.groupby('Club').apply(
            lambda g: (g['Goals_per90']*0.65 + g['Assists_per90']*0.35).mean()
        )

    if 'Defender' in pos_data:
        sub = pos_data['Defender']
        club_scores['Defensive'] = sub.groupby('Club').apply(
            lambda g: (
                g['TackleSuccessRate']/100 * 0.25 +
                g['Tackles_per90']         * 0.20 +
                g['Interceptions']         * 0.18 +
                g['Clearances']            * 0.12 +
                g['AerialDuelsWon']        * 0.13 +
                g['Recoveries']            * 0.12
            ).mean()
        )

    if 'Midfielder' in pos_data:
        sub = pos_data['Midfielder']
        club_scores['Midfield'] = sub.groupby('Club').apply(
            lambda g: (
                g['KeyPasses_per90']        * 0.35 +
                g['ProgressiveCarries']     * 0.28 +
                g['Touches_in_PenaltyArea'] * 0.22 +
                g['NonPenaltyGoals_per90']  * 0.15
            ).mean()
        )

    if 'Goalkeeper' in pos_data:
        sub = pos_data['Goalkeeper']
        club_scores['GK'] = sub.groupby('Club').apply(
            lambda g: (
                g['xSaves_per90']     * 0.50 +
                g['PostShot_xG']      * 0.30 +
                g['ShotsFaced_per90'] * 0.20
            ).mean()
        )

    combined = pd.DataFrame(club_scores).fillna(0)
    combined['Overall'] = combined.mean(axis=1)
    combined = combined.sort_values('Overall', ascending=False)

    print("\n  Club Scores:")
    print(combined.round(4).to_string())
    combined.to_csv(os.path.join(OUT_DIR, 'club_match_scores.csv'))

    _plot_match_analysis(combined)
    return combined


def _plot_match_analysis(scores: pd.DataFrame):
    score_cols = [c for c in scores.columns if c != 'Overall']
    clubs      = scores.index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), facecolor=C_BG)
    fig.suptitle('I-League — Club Match Analysis',
                 color=C_TEXT, fontsize=15, fontweight='bold')

    # Heatmap
    ax1.set_facecolor(C_CARD)
    sns.heatmap(scores[score_cols].T, ax=ax1, cmap='YlOrRd',
                annot=True, fmt='.3f', linewidths=0.5, linecolor=C_BG,
                annot_kws={'size': 9, 'color': 'white'},
                cbar_kws={'shrink': 0.75})
    ax1.set_title('Performance Heatmap', color=C_TEXT, fontsize=13, fontweight='bold')
    ax1.tick_params(colors=C_MUTED, labelsize=8)

    # Grouped bar chart
    ax2.set_facecolor(C_CARD)
    for sp in ax2.spines.values(): sp.set_edgecolor(C_BORDER)
    ax2.tick_params(colors=C_MUTED, labelsize=8)

    palette = [C_SELECT, '#00c8ff', C_WATCH, POS_COLOR['Goalkeeper']]
    x = np.arange(len(clubs)); w = 0.18
    for i, (col, clr) in enumerate(zip(score_cols, palette)):
        offset = (i - len(score_cols)/2 + 0.5) * w
        ax2.bar(x + offset, scores[col], width=w,
                label=col, color=clr, alpha=0.85, edgecolor=C_BG)
    ax2.set_xticks(x)
    ax2.set_xticklabels(clubs, rotation=40, ha='right', fontsize=8)
    ax2.set_title('Score Breakdown by Club', color=C_TEXT, fontsize=13, fontweight='bold')
    ax2.legend(facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT, fontsize=9)

    path = os.path.join(OUT_DIR, 'match_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print(f"\n  📊 Match analysis  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — MATCH OUTCOME PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_match_outcomes(club_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Synthesise head-to-head features from club score differentials.
    GradientBoosting predicts Win / Draw / Loss.
    WHY this algorithm? It handles multi-class tabular data well and is robust
    when feature count is low relative to samples.
    """
    print("\n" + "═"*62)
    print("  MODULE 6 — MATCH OUTCOME PREDICTION")
    print("═"*62)

    score_cols = [c for c in club_scores.columns if c != 'Overall']
    clubs      = club_scores.index.tolist()

    if len(clubs) < 2:
        print("  ⚠  Need ≥2 clubs."); return pd.DataFrame()

    rows = []
    for a, b in itertools.permutations(clubs, 2):
        feat = {f'{c}_diff': club_scores.loc[a, c] - club_scores.loc[b, c]
                for c in score_cols}
        feat['Overall_diff'] = club_scores.loc[a,'Overall'] - club_scores.loc[b,'Overall']
        adv = feat['Overall_diff']
        feat.update({'Home': a, 'Away': b,
                     'Outcome': 'Win' if adv > 0.05 else 'Loss' if adv < -0.05 else 'Draw'})
        rows.append(feat)

    mdf      = pd.DataFrame(rows)
    feat_c   = [c for c in mdf.columns if c.endswith('_diff')]
    X        = mdf[feat_c].values
    le       = LabelEncoder()
    y        = le.fit_transform(mdf['Outcome'])

    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.10,
                                     max_depth=3, random_state=42)
    if len(set(y)) >= 2:
        n_sp = max(2, min(5, int(np.bincount(y).min())))
        cv   = StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=42)
        accs = cross_val_score(gb, X, y, cv=cv, scoring='accuracy')
        print(f"  GradientBoosting CV Accuracy: {accs.mean():.3f}  (±{accs.std():.3f})")
        gb.fit(X, y)
        mdf['Predicted'] = le.inverse_transform(gb.predict(X))
    else:
        mdf['Predicted'] = mdf['Outcome']

    out = mdf[['Home','Away','Outcome','Predicted']]
    out.to_csv(os.path.join(OUT_DIR, 'match_outcomes.csv'), index=False)

    print("\n  Sample predictions (first 12 matches):")
    print(out.head(12).to_string(index=False))

    _plot_outcomes(out)
    return out


def _plot_outcomes(mdf: pd.DataFrame):
    oc_map = {'Win': C_SELECT, 'Draw': C_WATCH, 'Loss': POS_COLOR['Goalkeeper']}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=C_BG)
    for ax, col, title in zip(axes, ['Outcome','Predicted'],
                               ['Rule-Based Actual','Model Predicted']):
        ax.set_facecolor(C_CARD)
        cnts   = mdf[col].value_counts()
        colors = [oc_map.get(c, C_MUTED) for c in cnts.index]
        _, _, autotexts = ax.pie(
            cnts.values, labels=cnts.index, colors=colors,
            autopct='%1.0f%%', startangle=140,
            wedgeprops={'edgecolor': C_BG, 'linewidth': 2},
            textprops={'color': C_TEXT, 'fontsize': 11})
        for at in autotexts: at.set_color(C_BG); at.set_fontweight('bold')
        ax.set_title(title, color=C_TEXT, fontsize=12, fontweight='bold')
    fig.suptitle('Match Outcome Distribution', color=C_TEXT, fontsize=14, fontweight='bold')
    path = os.path.join(OUT_DIR, 'match_outcomes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print(f"\n  📊 Outcomes chart  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 7 — TACTICAL RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def tactical_recommendations(pos_data: dict, club_scores: pd.DataFrame) -> list:
    """Rule-based tactical advice derived from each club's score profile."""
    print("\n" + "═"*62)
    print("  MODULE 7 — TACTICAL RECOMMENDATIONS")
    print("═"*62)

    score_cols = [c for c in club_scores.columns if c != 'Overall']
    col_mean   = club_scores[score_cols].mean()
    recs       = []

    for club in club_scores.index:
        row    = club_scores.loc[club, score_cols]
        str_   = row[row >= col_mean].index.tolist()
        weak   = row[row <  col_mean].index.tolist()
        advice = []

        if 'Defensive' in weak:
            advice.append("Scout high-tackle% defenders (Tackle Success ≥ 70 %)")
        if 'Attacking' in weak:
            advice.append("Recruit finishers (Goals/90 ≥ 0.46)")
        if 'Midfield' in weak:
            advice.append("Target creative MFs (Key Passes ≥ 2.2, Prog.Carries ≥ 7)")
        if 'GK' in weak:
            advice.append("Upgrade goalkeeper (xSaves/90 ≥ 3.5)")
        if not advice:
            advice.append("Well-rounded — focus on squad depth")

        formation = _suggest_formation(str_, weak)
        rec = {
            'Club':       club,
            'Overall':    round(club_scores.loc[club, 'Overall'], 4),
            'Strengths':  ', '.join(str_) or '—',
            'Weaknesses': ', '.join(weak) or '—',
            'Formation':  formation,
            'Advice':     ' | '.join(advice),
        }
        recs.append(rec)
        print(f"\n  🏟  {club}  (Overall={rec['Overall']})")
        print(f"      Formation : {formation}")
        print(f"      Strengths : {rec['Strengths']}")
        print(f"      Weaknesses: {rec['Weaknesses']}")
        for a in advice: print(f"      → {a}")

    pd.DataFrame(recs).to_csv(
        os.path.join(OUT_DIR, 'tactical_recommendations.csv'), index=False)
    _plot_tactical_radars(club_scores, score_cols)
    return recs


def _suggest_formation(strengths, weaknesses):
    has = lambda x: x in strengths
    if has('Attacking') and has('Midfield'):  return '4-3-3  (Attack-focused)'
    if has('Defensive') and has('Midfield'):  return '4-5-1  (Defensive Midfield)'
    if has('Defensive'):                       return '5-3-2  (Compact Defence)'
    if has('Midfield'):                        return '4-2-3-1 (Midfield Dominance)'
    return '4-4-2  (Balanced)'


def _plot_tactical_radars(club_scores: pd.DataFrame, score_cols: list):
    clubs   = club_scores.index.tolist()
    N       = len(score_cols)
    angles  = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    n_cols  = 3
    n_rows  = (len(clubs) + n_cols - 1) // n_cols
    palette = plt.cm.tab10(np.linspace(0, 1, len(clubs)))

    fig = plt.figure(figsize=(6*n_cols, 5*n_rows), facecolor=C_BG)
    fig.suptitle('Tactical Radar — Club Profiles',
                 color=C_TEXT, fontsize=16, fontweight='bold')

    mn = club_scores[score_cols].min()
    mx = club_scores[score_cols].max()

    for i, (club, clr) in enumerate(zip(clubs, palette)):
        ax = fig.add_subplot(n_rows, n_cols, i+1, polar=True)
        ax.set_facecolor(C_CARD)
        ax.spines['polar'].set_color(C_BORDER)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(score_cols, color=C_MUTED, fontsize=9)

        norm = [(club_scores.loc[club, s] - mn[s]) / (mx[s] - mn[s] + 1e-9)
                for s in score_cols]
        norm += norm[:1]

        ax.plot(angles, norm, color=clr, linewidth=2)
        ax.fill(angles, norm, color=clr, alpha=0.25)
        ax.set_ylim(0, 1); ax.yaxis.set_ticklabels([])
        ax.set_title(club, color=C_TEXT, fontsize=10, fontweight='bold', pad=16)

    path = os.path.join(OUT_DIR, 'tactical_radars.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print(f"\n  📊 Tactical radars  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 8 — PLAYER RADAR CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def player_radar_charts(pos_data: dict):
    """Generate radar profiles for the top 4 players in each position."""
    print("\n" + "═"*62)
    print("  MODULE 8 — PLAYER RADAR CHARTS")
    print("═"*62)

    for pos, sub in pos_data.items():
        cfg   = POS_CONFIG[pos]
        mets  = cfg['metrics']
        lbls  = cfg['labels']
        color = POS_COLOR[pos]
        N     = len(mets)
        angles= np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

        top = sub.nlargest(min(4, len(sub)), 'Composite_Score')
        if top.empty: continue

        fig = plt.figure(figsize=(5.5*len(top), 5.5), facecolor=C_BG)
        fig.suptitle(f'{pos}  —  Top Player Radar Profiles',
                     color=color, fontsize=14, fontweight='bold')

        mn = sub[mets].min(); mx = sub[mets].max()

        for i, (_, player) in enumerate(top.iterrows()):
            ax = fig.add_subplot(1, len(top), i+1, polar=True)
            ax.set_facecolor(C_CARD)
            ax.spines['polar'].set_color(C_BORDER)
            ax.tick_params(colors=C_MUTED, labelsize=7.5)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(lbls, color=C_MUTED, fontsize=8)

            # standard ring (dashed white)
            std_norm = [(cfg['standards'][m] - mn[m]) / (mx[m] - mn[m] + 1e-9)
                        for m in mets]
            std_norm = [np.clip(v, 0, 1) for v in std_norm] + std_norm[:1]
            ax.plot(angles, std_norm, color='white', lw=1, ls='--', alpha=0.4)

            # player polygon
            vals = [np.clip((player[m] - mn[m]) / (mx[m] - mn[m] + 1e-9), 0, 1)
                    for m in mets]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, lw=2.2)
            ax.fill(angles, vals, color=color, alpha=0.28)
            ax.set_ylim(0, 1); ax.yaxis.set_ticklabels([])

            vc = VERDICT_COLOR.get(player['Label'], C_TEXT)
            ax.set_title(
                f"{player['Name']}\n{player['Club']}\n"
                f"Score: {player['Composite_Score']:.3f}  [{player['Label']}]",
                color=vc, fontsize=8.5, pad=20)

        path = os.path.join(OUT_DIR, f'radar_{pos.lower()}.png')
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=C_BG)
        plt.close()
        print(f"  📊 Radar  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 9 — MASTER SCOUT SHORTLIST EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_scout_shortlist(pos_data: dict):
    """Save a combined ranked shortlist of all SELECT/WATCH players to CSV."""
    print("\n" + "═"*62)
    print("  MODULE 9 — SCOUT SHORTLIST EXPORT")
    print("═"*62)

    rows = []
    for pos, sub in pos_data.items():
        shortlist = sub[sub['Label'].isin(['SELECT','WATCH'])].copy()
        shortlist = shortlist.nlargest(len(shortlist), 'Composite_Score')
        shortlist.insert(0, 'Rank', range(1, len(shortlist)+1))
        rows.append(shortlist[['Rank','Name','Position','Club',
                                'Composite_Score','Label'] +
                               POS_CONFIG[pos]['metrics']])

    full = pd.concat(rows, ignore_index=True)
    path = os.path.join(OUT_DIR, 'scout_shortlist.csv')
    full.to_csv(path, index=False)
    print(f"\n  {'Rank':<5} {'Name':<25} {'Position':<12} {'Club':<25} {'Score':<8} Verdict")
    print(f"  {'─'*5} {'─'*25} {'─'*12} {'─'*25} {'─'*8} {'─'*7}")
    for _, r in full.iterrows():
        print(f"  {int(r['Rank']):<5} {r['Name']:<25} {r['Position']:<12} "
              f"{r['Club']:<25} {r['Composite_Score']:.4f}  {r['Label']}")
    print(f"\n  📄 Scout shortlist  →  {path}")
    return full


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT NEW PLAYER  (utility — call after training)
# ══════════════════════════════════════════════════════════════════════════════

def predict_player(metrics: dict, pos: str) -> dict:
    """
    Load saved model and return scout verdict for a new player.

    Args
    ----
    metrics : dict  e.g. {'Goals_per90': 0.52, 'Assists_per90': 0.21}
    pos     : 'Goalkeeper' | 'Defender' | 'Midfielder' | 'Attacker'

    Returns
    -------
    dict with 'verdict', 'probabilities', 'composite_score'

    Example
    -------
    >>> result = predict_player({'Goals_per90': 0.55, 'Assists_per90': 0.25}, 'Attacker')
    >>> print(result)
    """
    pkl = os.path.join(OUT_DIR, f'model_{pos.lower()}.pkl')
    if not os.path.exists(pkl):
        raise FileNotFoundError(f"No saved model for {pos}. Run main() first.")

    pkg = joblib.load(pkl)
    mdl, le = pkg['model'], pkg['le']
    cfg     = POS_CONFIG[pos]

    # Build a single-row DataFrame
    row = pd.DataFrame([metrics])

    # Add engineered features (can't normalise from 1 row → use 0.5 placeholder)
    for m in cfg['metrics']:
        row[f'{m}_vs_std'] = (row.get(m, 0) - cfg['standards'][m]) * (
            1 if cfg['higher_better'][m] else -1)
        row[f'{m}_norm'] = 0.5

    row['composite'] = 0.5   # placeholder
    expected = pkg['feature_cols']
    for c in expected:
        if c not in row.columns: row[c] = 0.0
    X = row[expected]

    verdict = le.inverse_transform(mdl.predict(X))[0]
    try:
        proba = {le.classes_[i]: round(float(p), 3)
                 for i, p in enumerate(mdl.predict_proba(X)[0])}
    except Exception:
        proba = {verdict: 1.0}

    return {'verdict': verdict, 'probabilities': proba}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main(xlsx_path: str = DATA_FILE):
    t0 = datetime.now()
    print("\n" + "╔" + "═"*60 + "╗")
    print("║   SCOUT VISION — I-League Football Analytics System      ║")
    print("╚" + "═"*60 + "╝")

    # ── 1. Load
    df = load_xlsx(xlsx_path)

    # ── 2. Preprocess
    pos_data = preprocess(df)

    # ── 3. Performance report
    performance_report(pos_data)

    # ── 4. Similarity
    player_similarity(pos_data, k=5)

    # ── 5. Selection models
    model_store = train_selection_models(pos_data)

    # ── 6. Match analysis
    club_scores = match_analysis(pos_data)

    # ── 7. Outcome prediction
    predict_match_outcomes(club_scores)

    # ── 8. Tactical recommendations
    tactical_recommendations(pos_data, club_scores)

    # ── 9. Radar charts
    player_radar_charts(pos_data)

    # ── 10. Export shortlist
    export_scout_shortlist(pos_data)

    # ── Demo: predict a new unseen player
    print("\n" + "═"*62)
    print("  DEMO — predict_player() on unseen data")
    print("═"*62)
    demos = [
        ('Attacker',   {'Goals_per90': 0.55, 'Assists_per90': 0.24},   "Strong finisher"),
        ('Attacker',   {'Goals_per90': 0.21, 'Assists_per90': 0.06},   "Weak attacker"),
        ('Defender',   {'Tackles_per90': 2.60, 'TackleSuccessRate': 73.0,
                        'Interceptions': 2.10, 'Clearances': 5.80,
                        'AerialDuelsWon': 3.40, 'Recoveries': 7.50},   "Elite CB"),
        ('Goalkeeper', {'xSaves_per90': 3.75, 'PostShot_xG': 0.47,
                        'ShotsFaced_per90': 5.40},                      "Top GK"),
        ('Midfielder', {'KeyPasses_per90': 2.50, 'ProgressiveCarries': 7.40,
                        'Touches_in_PenaltyArea': 5.20,
                        'NonPenaltyGoals_per90': 0.17},                 "Creative CM"),
    ]
    for pos, metrics, desc in demos:
        res  = predict_player(metrics, pos)
        icon = {'SELECT':'✅','WATCH':'👀','PASS':'❌'}.get(res['verdict'],'❓')
        print(f"\n  {icon}  [{pos}] {desc}")
        print(f"      Verdict  : {res['verdict']}")
        prob_str = '  '.join(f"{k}={v:.0%}" for k,v in res['probabilities'].items())
        print(f"      Probs    : {prob_str}")

    elapsed = (datetime.now() - t0).seconds
    print("\n" + "╔" + "═"*60 + "╗")
    print(f"║  ✅  Complete in {elapsed}s  │  outputs/ → {len(os.listdir(OUT_DIR))} files  ║")
    print("╚" + "═"*60 + "╝\n")

    print("  Generated files:")
    for f in sorted(os.listdir(OUT_DIR)):
        kb = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
        print(f"    {f:<46}  {kb:>5} KB")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_FILE
    main(path)
