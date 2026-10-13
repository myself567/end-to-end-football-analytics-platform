"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     END TO END FOOTBALL ANALYTICS — Streamlit GUI                           ║
║         I-League Player Selection System                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install streamlit pandas openpyxl scikit-learn matplotlib seaborn numpy joblib
    streamlit run scout_vision.py

Place in same folder as:
    - i_league_players_dataset.xlsx
    - scout_ml.py  (optional — for running full pipeline)
    - outputs/     (optional — for pre-generated charts)
"""

import os
import io
import math
import warnings
import itertools

import numpy  as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.neighbors       import NearestNeighbors
from sklearn.decomposition   import PCA
from sklearn.ensemble        import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics         import classification_report
from sklearn.pipeline        import Pipeline

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="End to End Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES  (green + red theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Source+Sans+3:wght@300;400;600&display=swap');

/* ══════════════════════════════════════════════════════════
   ROOT BACKGROUNDS
   ══════════════════════════════════════════════════════════ */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0d2818 0%, #1a4d2e 50%, #0d2818 100%) !important;
    font-family: 'Source Sans 3', sans-serif;
}
[data-testid="stMain"] { background: transparent !important; }
.block-container {
    padding: 1.2rem 2rem 2rem 2rem !important;
    max-width: 1200px !important;
}

/* ══════════════════════════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1f10 0%, #1a3a20 100%) !important;
    border-right: 2px solid #c0392b !important;
}
[data-testid="stSidebar"] * { color: #f0fff4 !important; }
[data-testid="stSidebar"] .stRadio label  { color: #a8d5ba !important; font-size: 0.95rem; }
[data-testid="stSidebar"] .stSelectbox label { color: #a8d5ba !important; }

/* ══════════════════════════════════════════════════════════
   HEADINGS
   ══════════════════════════════════════════════════════════ */
h1, h2, h3 { font-family: 'Oswald', sans-serif !important; letter-spacing: 1px; }
h1 { color: #ffffff !important; }
h2 { color: #f4c430 !important; }
h3 { color: #7cfc00 !important; }

/* ══════════════════════════════════════════════════════════
   METRIC CARDS
   ══════════════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: rgba(26, 77, 46, 0.7) !important;
    border: 1px solid #40916c !important;
    border-radius: 10px !important;
    padding: 12px !important;
}
[data-testid="metric-container"] label { color: #a8d5ba !important; font-size: 0.8rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f4c430 !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 1.8rem !important;
}

/* ══════════════════════════════════════════════════════════
   DATAFRAME
   ══════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ══════════════════════════════════════════════════════════
   SELECTBOX — CLOSED STATE (the visible box with selected value)
   ══════════════════════════════════════════════════════════ */
/* Outer wrapper */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > label + div > div {
    background-color: #0f3320 !important;
    border: 1.5px solid #40916c !important;
    border-radius: 7px !important;
}
/* The inner div that BaseWeb renders as the control */
[data-baseweb="select"] > div {
    background-color: #0f3320 !important;
    border: 1.5px solid #40916c !important;
    border-radius: 7px !important;
    color: #ffffff !important;
}
/* The selected-value text span inside the control */
[data-baseweb="select"] > div > div,
[data-baseweb="select"] span,
[data-baseweb="select"] [class*="placeholder"],
[data-baseweb="select"] [class*="singleValue"],
[data-baseweb="select"] [class*="value"] {
    color: #ffffff !important;
    font-weight: 600 !important;
}
/* Arrow/caret icon area */
[data-baseweb="select"] > div > div:last-child svg {
    fill: #7cfc00 !important;
}

/* ══════════════════════════════════════════════════════════
   SELECTBOX — OPEN DROPDOWN POPUP (the floating list)
   ══════════════════════════════════════════════════════════ */
[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"],
[data-baseweb="popover"] > div,
[class*="menuList"],
[class*="dropdown"] {
    background-color: #0f3320 !important;
    border: 1.5px solid #40916c !important;
    border-radius: 8px !important;
}
/* Each option row */
[data-baseweb="menu"] li,
[data-baseweb="option"],
[role="option"],
ul[role="listbox"] li {
    background-color: #0f3320 !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    padding: 8px 14px !important;
}
/* Hovered option */
[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover,
[role="option"]:hover,
ul[role="listbox"] li:hover {
    background-color: #1a5c38 !important;
    color: #f4c430 !important;
    cursor: pointer !important;
}
/* Currently selected / highlighted option */
[aria-selected="true"],
[data-baseweb="option"][aria-selected="true"] {
    background-color: #c0392b !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ══════════════════════════════════════════════════════════
   TEXT INPUT & NUMBER INPUT
   ══════════════════════════════════════════════════════════ */
.stTextInput input,
.stNumberInput input,
input[type="text"],
input[type="number"] {
    background-color: #0f3320 !important;
    color: #ffffff !important;
    border: 1.5px solid #40916c !important;
    border-radius: 7px !important;
    font-size: 0.95rem !important;
    caret-color: #7cfc00 !important;
}
.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: #7aad90 !important;
    opacity: 1 !important;
}
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #f4c430 !important;
    box-shadow: 0 0 0 2px rgba(244,196,48,0.25) !important;
    outline: none !important;
}
/* Labels above inputs */
.stTextInput label, .stNumberInput label,
.stSelectbox label, .stMultiSelect label {
    color: #a8d5ba !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}

/* ══════════════════════════════════════════════════════════
   FILE UPLOADER  ← fixes the top-left circled area
   ══════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background-color: #0f3320 !important;
    border: 2px dashed #40916c !important;
    border-radius: 10px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploader"] section {
    background-color: #0f3320 !important;
    border: none !important;
}
[data-testid="stFileUploader"] button {
    background-color: #1a5c38 !important;
    color: #ffffff !important;
    border: 1px solid #40916c !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #a8d5ba !important;
}
/* The drag-drop zone */
[data-testid="stFileUploaderDropzone"] {
    background-color: #0f3320 !important;
    border: 2px dashed #40916c !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #ffffff !important;
}
/* Browse files button inside uploader */
[data-testid="baseButton-secondary"] {
    background-color: #1a5c38 !important;
    color: #ffffff !important;
    border: 1px solid #52b788 !important;
    border-radius: 6px !important;
}
/* Uploaded file name pill */
[data-testid="stFileUploaderFile"] {
    background-color: #122a1c !important;
    border: 1px solid #40916c !important;
    border-radius: 6px !important;
    color: #ffffff !important;
}
[data-testid="stFileUploaderFileName"] {
    color: #7cfc00 !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #c0392b, #922b21) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    padding: 0.5rem 1.5rem !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e74c3c, #c0392b) !important;
    box-shadow: 0 4px 12px rgba(192,57,43,0.4) !important;
}

/* ══════════════════════════════════════════════════════════
   TABS
   ══════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10, 31, 16, 0.8) !important;
    border-radius: 10px 10px 0 0 !important;
    border-bottom: 2px solid #c0392b !important;
    gap: 4px !important;
    padding: 4px 8px 0 8px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #a8d5ba !important;
    font-family: 'Oswald', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(180deg, #c0392b, #922b21) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(10, 31, 16, 0.6) !important;
    border-radius: 0 0 10px 10px !important;
    padding: 1.2rem !important;
    border: 1px solid #2d6a4f !important;
    border-top: none !important;
}

/* ══════════════════════════════════════════════════════════
   EXPANDER
   ══════════════════════════════════════════════════════════ */
[data-testid="stExpander"] > details > summary {
    background: rgba(26, 58, 32, 0.85) !important;
    border: 1px solid #40916c !important;
    border-radius: 8px !important;
    color: #f4c430 !important;
    font-family: 'Oswald', sans-serif !important;
    padding: 10px 14px !important;
}
[data-testid="stExpander"] > details > summary span {
    color: #f4c430 !important;
}
[data-testid="stExpander"] > details[open] > summary {
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stExpander"] > details > div {
    background: rgba(13, 40, 24, 0.85) !important;
    border: 1px solid #2d6a4f !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 12px !important;
}

/* ══════════════════════════════════════════════════════════
   RADIO BUTTONS
   ══════════════════════════════════════════════════════════ */
.stRadio > div { gap: 6px !important; }
.stRadio label {
    color: #e0f0e8 !important;
    font-size: 0.92rem !important;
}
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #e0f0e8 !important;
}

/* ══════════════════════════════════════════════════════════
   PROGRESS BAR
   ══════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div > div {
    background-color: #40916c !important;
}
[data-testid="stProgress"] > div {
    background-color: #122a1c !important;
    border-radius: 99px !important;
}

/* ══════════════════════════════════════════════════════════
   ALERT / INFO / WARNING BOXES
   ══════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    background-color: #0f3320 !important;
    border-radius: 8px !important;
    border-left: 4px solid #40916c !important;
    color: #e0f0e8 !important;
}
[data-testid="stAlert"] * { color: #e0f0e8 !important; }

/* ══════════════════════════════════════════════════════════
   VERDICT BADGES
   ══════════════════════════════════════════════════════════ */
.badge-select { background:#f4c430; color:#0d2818; padding:3px 10px; border-radius:99px; font-weight:700; font-size:0.78rem; }
.badge-watch  { background:#e09c00; color:#fff;    padding:3px 10px; border-radius:99px; font-weight:700; font-size:0.78rem; }
.badge-pass   { background:#c0392b; color:#fff;    padding:3px 10px; border-radius:99px; font-weight:700; font-size:0.78rem; }

/* ══════════════════════════════════════════════════════════
   CARD PANELS
   ══════════════════════════════════════════════════════════ */
.info-card {
    background: rgba(26, 77, 46, 0.5);
    border: 1px solid #40916c;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.red-card {
    background: rgba(192, 57, 43, 0.2);
    border: 1px solid #c0392b;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

/* ══════════════════════════════════════════════════════════
   DIVIDER & GENERAL TEXT
   ══════════════════════════════════════════════════════════ */
hr { border-color: #2d6a4f !important; }
p, li { color: #e0f0e8; }
/* Prevent generic div rule from overriding widget internals */
div:not([data-baseweb]):not([class*="base-"]) { color: #e0f0e8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# POSITION CONFIG
# ─────────────────────────────────────────────────────────────────────────────
POS_CONFIG = {
    'Goalkeeper': {
        'metrics':       ['xSaves_per90', 'PostShot_xG', 'ShotsFaced_per90'],
        'labels':        ['xSaves/90',    'PostShot xG',  'Shots Faced/90'],
        'standards':     {'xSaves_per90': 3.50, 'PostShot_xG': 0.44, 'ShotsFaced_per90': 5.20},
        'higher_better': {'xSaves_per90': True, 'PostShot_xG': True, 'ShotsFaced_per90': True},
        'weights':       {'xSaves_per90': 0.50, 'PostShot_xG': 0.30, 'ShotsFaced_per90': 0.20},
        'color': '#ff6b35',
    },
    'Defender': {
        'metrics':       ['Tackles_per90','TackleSuccessRate','Interceptions',
                          'Clearances','AerialDuelsWon','Recoveries'],
        'labels':        ['Tackles/90','Tackle %','Interceptions',
                          'Clearances','Aerial/90','Recoveries'],
        'standards':     {'Tackles_per90':2.30,'TackleSuccessRate':70.0,'Interceptions':2.00,
                          'Clearances':5.50,'AerialDuelsWon':3.20,'Recoveries':7.20},
        'higher_better': {m:True for m in ['Tackles_per90','TackleSuccessRate','Interceptions',
                          'Clearances','AerialDuelsWon','Recoveries']},
        'weights':       {'Tackles_per90':0.20,'TackleSuccessRate':0.25,'Interceptions':0.18,
                          'Clearances':0.15,'AerialDuelsWon':0.12,'Recoveries':0.10},
        'color': '#00c8ff',
    },
    'Midfielder': {
        'metrics':       ['KeyPasses_per90','ProgressiveCarries',
                          'Touches_in_PenaltyArea','NonPenaltyGoals_per90'],
        'labels':        ['Key Passes/90','Prog. Carries','Box Touches','NP Goals/90'],
        'standards':     {'KeyPasses_per90':2.20,'ProgressiveCarries':7.00,
                          'Touches_in_PenaltyArea':5.00,'NonPenaltyGoals_per90':0.14},
        'higher_better': {m:True for m in ['KeyPasses_per90','ProgressiveCarries',
                          'Touches_in_PenaltyArea','NonPenaltyGoals_per90']},
        'weights':       {'KeyPasses_per90':0.35,'ProgressiveCarries':0.28,
                          'Touches_in_PenaltyArea':0.22,'NonPenaltyGoals_per90':0.15},
        'color': '#7cfc00',
    },
    'Attacker': {
        'metrics':       ['Goals_per90','Assists_per90'],
        'labels':        ['Goals/90','Assists/90'],
        'standards':     {'Goals_per90':0.46,'Assists_per90':0.20},
        'higher_better': {'Goals_per90':True,'Assists_per90':True},
        'weights':       {'Goals_per90':0.65,'Assists_per90':0.35},
        'color': '#ffd700',
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGO  — drawn with matplotlib, returned as PNG bytes
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def build_logo_svg(width=700, height=160):
    """Draw the End to End Football Analytics header banner with footballer logo."""
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    ax.set_xlim(0, width); ax.set_ylim(0, height)
    ax.axis('off')

    # ── Pitch stripe background ──────────────────────────────────────────────
    stripe_w = 60
    for i in range(0, width + stripe_w, stripe_w):
        c = '#0d2818' if (i // stripe_w) % 2 == 0 else '#1a4d2e'
        ax.add_patch(plt.Rectangle((i, 0), stripe_w, height,
                                   color=c, zorder=0))

    # Border
    ax.add_patch(FancyBboxPatch((4, 4), width-8, height-8,
                                boxstyle="round,pad=2",
                                edgecolor='#40916c', facecolor='none',
                                linewidth=1.5, zorder=2))

    # ── Footballer figure ────────────────────────────────────────────────────
    ox, oy = 68, 82   # origin (player centre)
    s  = 1.1          # scale
    wh = 'white'

    def poly(*pts, color=wh, zorder=5):
        xs = [p[0]*s + ox for p in pts]
        ys = [p[1]*s + oy for p in pts]
        ax.fill(xs, ys, color=color, zorder=zorder)

    # Head
    ax.add_patch(plt.Circle((ox, oy + 36*s), 9*s, color=wh, zorder=5))
    # Torso
    poly((-8,18),(8,18),(9,-2),(-9,-2))
    # Left leg (planted)
    poly((-8,-2),(-2,-2),(-2,-26),(-8,-26))
    # Left foot
    poly((-14,-28),(-2,-28),(-2,-26),(-8,-26),(-14,-26))
    # Right leg kicking
    poly((2,-2),(8,-2),(22,-14),(18,-19))
    # Shin / foot forward
    poly((22,-14),(28,-8),(34,-12),(28,-20))
    # Left arm
    poly((-8,14),(-14,2),(-18,4),(-12,16))
    # Right arm swinging back
    poly((8,14),(16,4),(20,8),(12,18))

    # ── Football ─────────────────────────────────────────────────────────────
    bx, by, br = ox + 36*s, oy - 8*s, 9*s
    ax.add_patch(plt.Circle((bx, by), br, color=wh, zorder=5))
    ax.add_patch(plt.Circle((bx, by), br, color='none',
                             ec='#0d2818', linewidth=1.5, zorder=6))
    # Pentagon patch
    penta_x = [bx + br*0.5*math.cos(math.radians(90 + i*72)) for i in range(5)] + [bx + br*0.5*math.cos(math.radians(90))]
    penta_y = [by + br*0.5*math.sin(math.radians(90 + i*72)) for i in range(5)] + [by + br*0.5*math.sin(math.radians(90))]
    ax.fill(penta_x, penta_y, color='#0d2818', zorder=7)

    # ── Titles ───────────────────────────────────────────────────────────────
    ax.text(130, 112, 'END TO END', fontsize=26, fontweight='bold',
            color='#f4c430', fontfamily='DejaVu Sans', va='center', zorder=10)
    ax.text(130, 72,  'FOOTBALL ANALYTICS', fontsize=22, fontweight='bold',
            color='white', fontfamily='DejaVu Sans', va='center', zorder=10)
    ax.text(130, 36, 'I-League  ·  End to End Player Scouting & ML Analytics',
            fontsize=11, color='#7cfc00', style='italic',
            fontfamily='DejaVu Sans', va='center', zorder=10)



    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='none', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  — supports CSV and Excel with column name normalisation
# ─────────────────────────────────────────────────────────────────────────────

# Column name aliases: maps common CSV column name variations to canonical names
COL_ALIASES = {
    'xsaves_per90': 'xSaves_per90', 'expected_saves_per90': 'xSaves_per90',
    'xsaves': 'xSaves_per90', 'postshot_xg': 'PostShot_xG',
    'post_shot_xg': 'PostShot_xG', 'post_shot_expected_goals': 'PostShot_xG',
    'shotsfaced_per90': 'ShotsFaced_per90', 'shots_faced_per90': 'ShotsFaced_per90',
    'shots_faced': 'ShotsFaced_per90', 'tackles_per90': 'Tackles_per90',
    'tacklesuccessrate': 'TackleSuccessRate', 'tackle_success_rate': 'TackleSuccessRate',
    'tackle_%': 'TackleSuccessRate', 'interceptions_per90': 'Interceptions',
    'interceptions': 'Interceptions', 'clearances_per90': 'Clearances',
    'clearances': 'Clearances', 'aerialduels_per90': 'AerialDuelsWon',
    'aerial_duels_won': 'AerialDuelsWon', 'aerialduels': 'AerialDuelsWon',
    'aerial_duels_won_per90': 'AerialDuelsWon', 'recoveries_per90': 'Recoveries',
    'recoveries': 'Recoveries', 'keypasses_per90': 'KeyPasses_per90',
    'key_passes_per90': 'KeyPasses_per90', 'progressivecarries': 'ProgressiveCarries',
    'progressive_carries': 'ProgressiveCarries',
    'progressive_carries_per90': 'ProgressiveCarries',
    'touches_in_penalty_area': 'Touches_in_PenaltyArea',
    'touches_in_penaltyarea': 'Touches_in_PenaltyArea',
    'box_touches': 'Touches_in_PenaltyArea',
    'nonpenaltygoals_per90': 'NonPenaltyGoals_per90',
    'non_penalty_goals_per90': 'NonPenaltyGoals_per90',
    'np_goals_per90': 'NonPenaltyGoals_per90',
    'goals_per90': 'Goals_per90', 'assists_per90': 'Assists_per90',
    'name': 'Name', 'position': 'Position', 'club': 'Club',
    'matches': 'Matches', 'minutes': 'Minutes',
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename CSV columns to canonical names, strip whitespace, coerce numerics."""
    # Rename via alias map
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(' ', '_')
        if key in COL_ALIASES:
            rename_map[col] = COL_ALIASES[key]
    if rename_map:
        df = df.rename(columns=rename_map)

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': pd.NA, 'None': pd.NA, '': pd.NA})

    # Force numeric conversion on every known metric column
    all_metrics = []
    for cfg in POS_CONFIG.values():
        all_metrics.extend(cfg['metrics'])
    for col in set(all_metrics):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_data(file_bytes: bytes, file_name: str = '') -> tuple:
    """
    Load CSV or Excel into a normalised DataFrame.
    Returns (df, error_string).  error_string is None on success.
    No @st.cache_data — session_state caching used instead so
    re-uploading a CSV always re-processes cleanly.
    """
    ext = os.path.splitext(file_name)[1].lower() if file_name else ''
    buf = io.BytesIO(file_bytes)
    df  = None
    err = None

    try:
        if ext == '.csv':
            for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
                try:
                    buf.seek(0)
                    df = pd.read_csv(buf, encoding=enc, skipinitialspace=True)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                buf.seek(0)
                df = pd.read_csv(buf, encoding='latin-1', skipinitialspace=True)
        else:
            # Excel (.xlsx / .xls) — also fallback for unknown extensions
            try:
                buf.seek(0)
                df = pd.read_excel(buf, engine='openpyxl')
            except Exception:
                buf.seek(0)
                df = pd.read_excel(buf)

        df = normalise_columns(df)

    except Exception as e:
        return None, str(e)

    return df, err


def assign_label(row, pos):
    cfg = POS_CONFIG[pos]
    met = sum(
        1 for m in cfg['metrics']
        if pd.notna(row.get(m)) and (
            row[m] >= cfg['standards'][m] if cfg['higher_better'][m]
            else row[m] <= cfg['standards'][m])
    )
    ratio = met / len(cfg['metrics'])
    return 'SELECT' if ratio == 1.0 else 'WATCH' if ratio >= 0.6 else 'PASS'


def compute_composite(sub: pd.DataFrame, pos: str) -> pd.Series:
    cfg   = POS_CONFIG[pos]
    score = pd.Series(0.0, index=sub.index)
    for m in cfg['metrics']:
        mn, mx = sub[m].min(), sub[m].max()
        norm   = (sub[m] - mn) / (mx - mn + 1e-9)
        score += (norm if cfg['higher_better'][m] else 1-norm) * cfg['weights'][m]
    return score


def preprocess(df: pd.DataFrame) -> dict:
    pos_data = {}
    for pos, cfg in POS_CONFIG.items():
        sub = df[df['Position'] == pos].copy().dropna(subset=cfg['metrics'])
        if sub.empty:
            continue
        sub = sub.reset_index(drop=True)
        sub['Label']           = sub.apply(lambda r: assign_label(r, pos), axis=1)
        sub['Composite_Score'] = compute_composite(sub, pos)
        pos_data[pos] = sub
    return pos_data


def engineer_features(sub: pd.DataFrame, pos: str) -> pd.DataFrame:
    cfg  = POS_CONFIG[pos]
    mets = cfg['metrics']
    X    = sub[mets].copy().astype(float)
    for m in mets:
        diff = X[m] - cfg['standards'][m]
        X[f'{m}_vs_std'] = diff if cfg['higher_better'][m] else -diff
        mn, mx = X[m].min(), X[m].max()
        norm   = (X[m] - mn) / (mx - mn + 1e-9)
        X[f'{m}_norm'] = norm if cfg['higher_better'][m] else 1-norm
    X['composite'] = sum(X[f'{m}_norm'] * cfg['weights'][m] for m in mets)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
DARK  = '#0d2818'
CARD  = '#1a4d2e'
VMAP  = {'SELECT':'#f4c430', 'WATCH':'#e09c00', 'PASS':'#c0392b'}

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_radar(player: pd.Series, pos: str) -> bytes:
    cfg    = POS_CONFIG[pos]
    mets   = cfg['metrics']
    labels = cfg['labels']
    N      = len(mets)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    sub_vals = [player.get(m, 0) or 0 for m in mets]
    fig, ax  = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True),
                             facecolor=DARK)
    ax.set_facecolor('#122a1c')
    ax.spines['polar'].set_color('#2d6a4f')
    ax.grid(color='#2d6a4f', linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, color='#a8d5ba')
    ax.yaxis.set_visible(False)

    # standards normalised
    all_vals_ref = {m: sub_vals[i] for i, m in enumerate(mets)}
    stds_norm = [0.5] * N   # midpoint placeholder
    stds_norm += stds_norm[:1]

    norm_vals = []
    for m, v in zip(mets, sub_vals):
        std = cfg['standards'][m]
        norm = v / (std * 1.5 + 1e-9)
        norm_vals.append(min(1, max(0, norm)))
    norm_vals += norm_vals[:1]

    col = VMAP.get(player.get('Label','PASS'), '#c0392b')
    ax.fill(angles, norm_vals, color=col, alpha=0.35)
    ax.plot(angles, norm_vals, color=col, linewidth=2.2)

    # dot on each vertex
    for a, v in zip(angles[:-1], norm_vals[:-1]):
        ax.plot(a, v, 'o', color=col, markersize=5)

    ax.set_title(f"{player.get('Name','Player')}\n"
                 f"{player.get('Club','')}",
                 color='white', fontsize=9, pad=14, fontweight='bold')
    return fig_to_bytes(fig)


def make_distribution(sub: pd.DataFrame, pos: str) -> bytes:
    cfg    = POS_CONFIG[pos]
    mets   = cfg['metrics']
    labels = cfg['labels']
    n      = len(mets)
    color  = cfg['color']

    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 4), facecolor=DARK)
    if n == 1: axes = [axes]

    for ax, m, lbl in zip(axes, mets, labels):
        ax.set_facecolor('#122a1c')
        for sp in ax.spines.values(): sp.set_color('#2d6a4f')
        ax.tick_params(colors='#a8d5ba', labelsize=7)

        parts = ax.violinplot(sub[m].dropna(), showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color); pc.set_alpha(0.4)
        parts['cmedians'].set_color(color)
        for k in ('cbars','cmins','cmaxes'): parts[k].set_color('#2d6a4f')

        jitter = np.random.uniform(-0.08, 0.08, len(sub))
        dot_c  = sub['Label'].map(VMAP)
        ax.scatter(1+jitter, sub[m], c=dot_c, s=22, alpha=0.85, zorder=4)

        std = cfg['standards'][m]
        ax.axhline(std, color='white', ls='--', lw=0.9, alpha=0.6)
        ax.text(1.38, std, f'{std}', color='white', fontsize=6, va='center')
        ax.set_title(lbl, color=color, fontsize=9, pad=4)
        ax.set_xticks([])
        ax.yaxis.label.set_color('#a8d5ba')

    fig.suptitle(f'{pos} — Metric Distributions',
                 color='white', fontsize=11, fontweight='bold', y=1.02)
    fig.patch.set_facecolor(DARK)
    plt.tight_layout()
    return fig_to_bytes(fig)


def make_similarity_pca(sub: pd.DataFrame, pos: str) -> bytes:
    cfg   = POS_CONFIG[pos]
    mets  = cfg['metrics']
    color = cfg['color']
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(sub[mets].astype(float))
    pca   = PCA(n_components=2, random_state=42)
    proj  = pca.fit_transform(X_sc)
    ev    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=DARK)
    ax.set_facecolor('#122a1c')
    for sp in ax.spines.values(): sp.set_color('#2d6a4f')
    ax.tick_params(colors='#a8d5ba', labelsize=8)

    for verdict, grp in sub.groupby('Label'):
        idx = grp.index
        ax.scatter(proj[idx, 0], proj[idx, 1],
                   c=VMAP[verdict], label=verdict,
                   s=75, alpha=0.85, edgecolors='white', linewidths=0.3)

    for i in sub[sub['Label']=='SELECT'].index:
        ax.annotate(sub.loc[i,'Name'],
                    (proj[i,0], proj[i,1]),
                    fontsize=7, color='white', alpha=0.9,
                    xytext=(5,5), textcoords='offset points')

    ax.set_title(f'{pos} — Similarity Map (PCA)', color='white',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', color='#a8d5ba')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', color='#a8d5ba')
    ax.legend(facecolor='#122a1c', edgecolor='#2d6a4f',
              labelcolor='white', fontsize=9)
    fig.patch.set_facecolor(DARK)
    plt.tight_layout()
    return fig_to_bytes(fig)


def make_top_players_bar(sub: pd.DataFrame, pos: str) -> bytes:
    color = POS_CONFIG[pos]['color']
    top   = sub.nlargest(min(12, len(sub)), 'Composite_Score')
    bar_c = [VMAP[l] for l in top['Label']]

    fig, ax = plt.subplots(figsize=(8, max(4, len(top)*0.45)), facecolor=DARK)
    ax.set_facecolor('#122a1c')
    for sp in ax.spines.values(): sp.set_color('#2d6a4f')
    ax.tick_params(colors='#a8d5ba', labelsize=8.5)

    bars = ax.barh(top['Name'], top['Composite_Score'],
                   color=bar_c, alpha=0.88, edgecolor=DARK, height=0.65)
    for bar, lbl in zip(bars, top['Label']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                lbl, va='center', color=VMAP[lbl], fontsize=8)

    ax.set_title(f'Top {len(top)} {pos}s by Composite Score',
                 color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('Composite Score', color='#a8d5ba')
    fig.patch.set_facecolor(DARK)
    plt.tight_layout()
    return fig_to_bytes(fig)


def make_club_heatmap(club_scores: pd.DataFrame) -> bytes:
    cols = [c for c in club_scores.columns if c != 'Overall']
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK)
    ax.set_facecolor(DARK)
    sns.heatmap(club_scores[cols].T, ax=ax, cmap='RdYlGn',
                annot=True, fmt='.3f', linewidths=0.4, linecolor=DARK,
                annot_kws={'size':8,'color':'white'},
                cbar_kws={'shrink':0.7})
    ax.set_title('Club Performance Heatmap', color='white',
                 fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors='#a8d5ba', labelsize=8)
    fig.patch.set_facecolor(DARK)
    plt.tight_layout()
    return fig_to_bytes(fig)


def make_club_bar(club_scores: pd.DataFrame) -> bytes:
    cols    = [c for c in club_scores.columns if c != 'Overall']
    clubs   = club_scores.index.tolist()
    palette = ['#ffd700','#00c8ff','#7cfc00','#ff6b35']
    x = np.arange(len(clubs)); w = 0.18

    fig, ax = plt.subplots(figsize=(11, 5), facecolor=DARK)
    ax.set_facecolor('#122a1c')
    for sp in ax.spines.values(): sp.set_color('#2d6a4f')
    ax.tick_params(colors='#a8d5ba', labelsize=7.5)

    for i, (col, clr) in enumerate(zip(cols, palette)):
        offset = (i - len(cols)/2 + 0.5)*w
        ax.bar(x+offset, club_scores[col], width=w,
               label=col, color=clr, alpha=0.85, edgecolor=DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(clubs, rotation=38, ha='right', fontsize=8)
    ax.set_title('Club Score Breakdown', color='white',
                 fontsize=12, fontweight='bold')
    ax.legend(facecolor='#122a1c', edgecolor='#2d6a4f',
              labelcolor='white', fontsize=9)
    fig.patch.set_facecolor(DARK)
    plt.tight_layout()
    return fig_to_bytes(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ML TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_models(pos_data: dict) -> dict:
    """Train selection models for each position."""
    store = {}
    for pos, sub in pos_data.items():
        X  = engineer_features(sub, pos)
        le = LabelEncoder()
        y  = le.fit_transform(sub['Label'])

        candidates = {
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08,
                max_depth=3, subsample=0.85, random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, class_weight='balanced', random_state=42),
            'Logistic Regression': Pipeline([
                ('sc', StandardScaler()),
                ('lr', LogisticRegression(C=1.0, class_weight='balanced',
                                           max_iter=1000, random_state=42)),
            ]),
        }

        min_cls  = int(np.bincount(y).min())
        n_splits = max(2, min(5, min_cls))
        cv       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        best_f1, best_name, best_mdl = -1, None, None
        results = {}
        for name, mdl in candidates.items():
            f1s  = cross_val_score(mdl, X, y, cv=cv, scoring='f1_weighted')
            mean = f1s.mean()
            results[name] = round(mean, 3)
            if mean > best_f1:
                best_f1, best_name, best_mdl = mean, name, mdl

        best_mdl.fit(X, y)
        store[pos] = {
            'model': best_mdl, 'le': le,
            'model_name': best_name, 'f1': best_f1,
            'cv_results': results, 'feature_cols': list(X.columns),
        }
    return store


def predict_player(metrics: dict, pos: str, model_store: dict) -> dict:
    if pos not in model_store:
        return {'verdict': '—', 'probabilities': {}}
    pkg = model_store[pos]
    mdl, le = pkg['model'], pkg['le']
    row = pd.DataFrame([metrics])
    cfg = POS_CONFIG[pos]
    for m in cfg['metrics']:
        row[f'{m}_vs_std'] = (row.get(m, 0) - cfg['standards'][m]) * (
            1 if cfg['higher_better'][m] else -1)
        row[f'{m}_norm'] = 0.5
    row['composite'] = 0.5
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


# ─────────────────────────────────────────────────────────────────────────────
# CLUB SCORES
# ─────────────────────────────────────────────────────────────────────────────
def compute_club_scores(pos_data: dict) -> pd.DataFrame:
    score_fns = {
        'Attacking': lambda df: (df.get('Goals_per90', 0)*0.65 + df.get('Assists_per90', 0)*0.35).mean()
            if 'Attacker' in pos_data and len(pos_data['Attacker']) else None,
        'Defensive': lambda df: None,
        'Midfield':  lambda df: None,
        'GK':        lambda df: None,
    }

    all_clubs = set()
    for sub in pos_data.values():
        all_clubs.update(sub['Club'].unique())
    all_clubs = sorted(all_clubs)

    rows = {}
    for club in all_clubs:
        row = {}
        # Attacking
        if 'Attacker' in pos_data:
            g = pos_data['Attacker'][pos_data['Attacker']['Club']==club]
            if len(g): row['Attacking'] = (g['Goals_per90']*0.65 + g['Assists_per90']*0.35).mean()
        # Defensive
        if 'Defender' in pos_data:
            g = pos_data['Defender'][pos_data['Defender']['Club']==club]
            if len(g):
                row['Defensive'] = (
                    g['TackleSuccessRate']/100*0.25 + g['Tackles_per90']*0.20 +
                    g['Interceptions']*0.18 + g['Clearances']*0.12 +
                    g['AerialDuelsWon']*0.13 + g['Recoveries']*0.12
                ).mean()
        # Midfield
        if 'Midfielder' in pos_data:
            g = pos_data['Midfielder'][pos_data['Midfielder']['Club']==club]
            if len(g):
                row['Midfield'] = (
                    g['KeyPasses_per90']*0.35 + g['ProgressiveCarries']*0.28 +
                    g['Touches_in_PenaltyArea']*0.22 + g['NonPenaltyGoals_per90']*0.15
                ).mean()
        # GK
        if 'Goalkeeper' in pos_data:
            g = pos_data['Goalkeeper'][pos_data['Goalkeeper']['Club']==club]
            if len(g):
                row['GK'] = (
                    g['xSaves_per90']*0.50 + g['PostShot_xG']*0.30 + g['ShotsFaced_per90']*0.20
                ).mean()
        if row:
            row['Overall'] = sum(row.values()) / len(row)
            rows[club] = row

    return pd.DataFrame(rows).T.fillna(0).sort_values('Overall', ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# TACTICAL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_tactics(club_scores: pd.DataFrame) -> list:
    cols     = [c for c in club_scores.columns if c != 'Overall']
    col_mean = club_scores[cols].mean()
    recs     = []
    for club in club_scores.index:
        row  = club_scores.loc[club, cols]
        str_ = row[row >= col_mean].index.tolist()
        weak = row[row <  col_mean].index.tolist()
        advice = []
        if 'Defensive' in weak: advice.append('🔴 Scout high-tackle% defenders (TSR ≥ 70%)')
        if 'Attacking' in weak: advice.append('🔴 Recruit finishers (Goals/90 ≥ 0.46)')
        if 'Midfield'  in weak: advice.append('🔴 Target creative MFs (KP ≥ 2.2, PC ≥ 7)')
        if 'GK'        in weak: advice.append('🔴 Upgrade goalkeeper (xSaves/90 ≥ 3.5)')
        if not advice:           advice.append('🟢 Well-rounded — focus on depth')
        has = lambda x: x in str_
        if has('Attacking') and has('Midfield'): form = '4-3-3 (Attack)'
        elif has('Defensive') and has('Midfield'): form = '4-5-1 (Def Mid)'
        elif has('Defensive'): form = '5-3-2 (Compact)'
        elif has('Midfield'):  form = '4-2-3-1 (Mid Dom)'
        else:                  form = '4-4-2 (Balanced)'
        recs.append({'Club': club, 'Overall': round(float(club_scores.loc[club,'Overall']),4),
                     'Formation': form,
                     'Strengths': ', '.join(str_) or '—',
                     'Weaknesses': ', '.join(weak) or '—',
                     'Advice': advice})
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# VERDICT BADGE HTML
# ─────────────────────────────────────────────────────────────────────────────
def verdict_badge(v):
    cls = {'SELECT':'badge-select','WATCH':'badge-watch','PASS':'badge-pass'}.get(v,'badge-pass')
    return f'<span class="{cls}">{v}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚽ End to End Football Analytics")
        st.markdown("---")

        uploaded = st.file_uploader(
            "📂 Upload Dataset (.xlsx or .csv)",
            type=['xlsx','xls','csv'],
            help="Upload i_league_players_dataset.xlsx / .csv or any compatible file")

        st.markdown("---")
        st.markdown("**📍 Navigation**")
        page = st.radio("", [
            "🏠 Overview",
            "👤 Players",
            "📊 Analytics",
            "🏆 Standings",
            "🗺 Tactics",
            "🔮 Predict",
        ], label_visibility='collapsed')

        st.markdown("---")
        st.markdown("**🎯 Analyst Standards**")
        pos_sel = st.selectbox("Position", list(POS_CONFIG.keys()),
                               label_visibility='visible')
        cfg = POS_CONFIG[pos_sel]
        for m, lbl in zip(cfg['metrics'], cfg['labels']):
            st.caption(f"**{lbl}** ≥ {cfg['standards'][m]}")

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.75rem;color:#52b788;text-align:center'>"
            "End to End Football Analytics<br>I-League System v2.0</div>",
            unsafe_allow_html=True)

    return uploaded, page


# ─────────────────────────────────────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(pos_data):
    st.markdown("## 🏠 Overview")
    total   = sum(len(v) for v in pos_data.values())
    selects = sum((v['Label']=='SELECT').sum() for v in pos_data.values())
    watches = sum((v['Label']=='WATCH').sum() for v in pos_data.values())
    passes  = sum((v['Label']=='PASS').sum() for v in pos_data.values())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Players", total)
    c2.metric("✅ SELECT", selects, help="Meets all standards")
    c3.metric("👀 WATCH",  watches, help="Meets 60%+ standards")
    c4.metric("❌ PASS",   passes,  help="Below threshold")

    st.markdown("---")
    cols = st.columns(4)
    for i, (pos, sub) in enumerate(pos_data.items()):
        with cols[i]:
            color = POS_CONFIG[pos]['color']
            vc    = sub['Label'].value_counts()
            st.markdown(
                f"<div class='info-card'>"
                f"<h4 style='color:{color};margin:0'>{pos}</h4>"
                f"<p style='font-size:1.6rem;font-family:Oswald;color:white;margin:4px 0'>{len(sub)}</p>"
                f"<small style='color:#a8d5ba'>SELECT: {vc.get('SELECT',0)}  "
                f"WATCH: {vc.get('WATCH',0)}  PASS: {vc.get('PASS',0)}</small>"
                f"</div>",
                unsafe_allow_html=True)

    # Summary shortlist
    st.markdown("### 🌟 Top Analyst Picks")
    rows = []
    for pos, sub in pos_data.items():
        top = sub.nlargest(3, 'Composite_Score')
        for _, r in top.iterrows():
            rows.append({'Position': pos, 'Name': r['Name'], 'Club': r['Club'],
                         'Score': round(r['Composite_Score'],4), 'Verdict': r['Label']})
    if rows:
        picks_df = pd.DataFrame(rows)
        st.dataframe(
            picks_df.style
            .applymap(lambda v: 'background-color:#1a4d2e;color:#f4c430;font-weight:bold'
                      if v == 'SELECT'
                      else ('background-color:#2d3a1a;color:#e09c00' if v == 'WATCH'
                            else 'background-color:#2d1a1a;color:#e74c3c' if v == 'PASS'
                            else ''),
                      subset=['Verdict'])
            .format({'Score': '{:.4f}'}),
            use_container_width=True, height=350)


def page_players(pos_data):
    st.markdown("## 👤 Player Explorer")
    col_pos, col_verdict, col_search = st.columns([2, 2, 3])

    with col_pos:
        pos = st.selectbox("Position", list(pos_data.keys()))
    with col_verdict:
        verdict_filter = st.selectbox("Verdict", ['ALL','SELECT','WATCH','PASS'])
    with col_search:
        search = st.text_input("🔍 Search player / club", "")

    if pos not in pos_data:
        st.warning("No data for this position.")
        return

    sub = pos_data[pos].copy()
    if verdict_filter != 'ALL':
        sub = sub[sub['Label'] == verdict_filter]
    if search:
        mask = (sub['Name'].str.contains(search, case=False, na=False) |
                sub['Club'].str.contains(search, case=False, na=False))
        sub  = sub[mask]

    sub_sorted = sub.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
    sub_sorted.insert(0, 'Rank', range(1, len(sub_sorted)+1))

    st.markdown(f"**{len(sub_sorted)} players shown**")

    display_cols = ['Rank','Name','Club','Composite_Score','Label'] + POS_CONFIG[pos]['metrics']
    show_cols    = [c for c in display_cols if c in sub_sorted.columns]

    def style_verdict(v):
        if v == 'SELECT': return 'background-color:#1a3a0a;color:#f4c430;font-weight:bold'
        if v == 'WATCH':  return 'background-color:#2a2a0a;color:#e09c00'
        if v == 'PASS':   return 'background-color:#2a0a0a;color:#e74c3c'
        return ''

    styled = (sub_sorted[show_cols]
              .style
              .applymap(style_verdict, subset=['Label'])
              .format({'Composite_Score': '{:.4f}'}, na_rep='—')
              .format({m: '{:.3f}' for m in POS_CONFIG[pos]['metrics']
                       if m in sub_sorted.columns}, na_rep='—'))
    st.dataframe(styled, use_container_width=True, height=420)

    # Player detail
    st.markdown("---")
    st.markdown("### 🔍 Player Detail & Radar")
    player_names = sub_sorted['Name'].tolist()
    if player_names:
        sel_name = st.selectbox("Select player for detail view", player_names)
        player_row = sub_sorted[sub_sorted['Name'] == sel_name].iloc[0]

        c1, c2 = st.columns([1, 1])
        with c1:
            verdict = player_row.get('Label','—')
            badge_html = verdict_badge(verdict)
            st.markdown(
                f"<div class='info-card'>"
                f"<h3 style='color:white;margin:0'>{player_row['Name']}  {badge_html}</h3>"
                f"<p style='color:#a8d5ba;margin:4px 0'>⚽ {pos}  &nbsp;·&nbsp; 🏟 {player_row.get('Club','—')}</p>"
                f"<p style='color:#f4c430;font-size:1.1rem'>Score: {player_row['Composite_Score']:.4f}</p>"
                f"</div>",
                unsafe_allow_html=True)
            st.markdown("**Metric Breakdown:**")
            cfg = POS_CONFIG[pos]
            for m, lbl in zip(cfg['metrics'], cfg['labels']):
                val = player_row.get(m)
                std = cfg['standards'][m]
                hb  = cfg['higher_better'][m]
                ok  = val is not None and not pd.isna(val) and (val >= std if hb else val <= std)
                icon = "🟢" if ok else "🔴"
                bar_val = min(1.0, float(val)/(std*1.5+1e-9)) if val is not None else 0
                st.markdown(
                    f"{icon} **{lbl}**: `{float(val):.3f}`  _(std: {std})_  "
                    f"{'✅ MEETS' if ok else '❌ BELOW'}",
                    unsafe_allow_html=False)
                st.progress(bar_val)

        with c2:
            radar_bytes = make_radar(player_row, pos)
            st.image(radar_bytes, use_container_width=True)


def page_analytics(pos_data):
    st.markdown("## 📊 Performance Analytics")
    pos = st.selectbox("Select Position", list(pos_data.keys()), key='analytics_pos')
    sub = pos_data[pos]

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🗺 Similarity Map", "🏅 Top Players"])

    with tab1:
        dist_bytes = make_distribution(sub, pos)
        st.image(dist_bytes, use_container_width=True)

        st.markdown("### Descriptive Statistics")
        cfg  = POS_CONFIG[pos]
        desc = sub[cfg['metrics']].describe().T.round(4)
        st.dataframe(desc, use_container_width=True)

    with tab2:
        sim_bytes = make_similarity_pca(sub, pos)
        st.image(sim_bytes, use_container_width=True)
        st.caption("PCA 2D projection coloured by scout verdict. "
                   "SELECT players are annotated.")

    with tab3:
        bar_bytes = make_top_players_bar(sub, pos)
        st.image(bar_bytes, use_container_width=True)


def page_standings(pos_data):
    st.markdown("## 🏆 Club Standings")

    with st.spinner("Computing club scores…"):
        club_scores = compute_club_scores(pos_data)

    # Leaderboard table
    display = club_scores.copy()
    display.insert(0, 'Rank', range(1, len(display)+1))
    display = display.round(4)

    medals = {1:'🥇', 2:'🥈', 3:'🥉'}
    display['Rank'] = [medals.get(i, str(i)) for i in display['Rank']]

    st.dataframe(
        display.style
        .background_gradient(subset=['Overall'], cmap='RdYlGn')
        .format({c: '{:.4f}' for c in club_scores.columns}),
        use_container_width=True, height=420)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🌡 Performance Heatmap")
        hm_bytes = make_club_heatmap(club_scores)
        st.image(hm_bytes, use_container_width=True)
    with c2:
        st.markdown("### 📊 Score Breakdown")
        bar_bytes = make_club_bar(club_scores)
        st.image(bar_bytes, use_container_width=True)


def page_tactics(pos_data):
    st.markdown("## 🗺 Tactical Recommendations")

    with st.spinner("Analysing club profiles…"):
        club_scores = compute_club_scores(pos_data)
        recs        = build_tactics(club_scores)

    for rec in recs:
        overall = rec['Overall']
        bar_c   = '#2d6a4f' if overall > 2 else '#1a4d2e'
        with st.expander(f"🏟  **{rec['Club']}**  — Overall: `{overall:.4f}`  |  Formation: `{rec['Formation']}`"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"✅ **Strengths:**  `{rec['Strengths']}`")
                st.markdown(f"⚠️ **Weaknesses:** `{rec['Weaknesses']}`")
                st.markdown(f"🔲 **Recommended Formation:** `{rec['Formation']}`")
            with c2:
                st.markdown("**Scout Advice:**")
                for line in rec['Advice']:
                    st.markdown(f"- {line}")


def page_predict(pos_data, model_store):
    st.markdown("## 🔮 Predict New Player")
    st.markdown("Enter a player's metrics to get an instant analyst verdict.")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        pos = st.selectbox("Position", list(POS_CONFIG.keys()), key='predict_pos')
        cfg = POS_CONFIG[pos]

        st.markdown("**Enter Metrics:**")
        metrics = {}
        for m, lbl in zip(cfg['metrics'], cfg['labels']):
            std = cfg['standards'][m]
            metrics[m] = st.number_input(
                f"{lbl}  _(standard: {std})_",
                min_value=0.0, max_value=200.0,
                value=float(std),
                step=0.01, format="%.3f",
                key=f'pred_{m}')

        predict_btn = st.button("⚽  GET ANALYST VERDICT", key='predict_btn')

    with c2:
        st.markdown("**Result:**")
        if predict_btn:
            # Rule-based
            met = sum(
                1 for m in cfg['metrics']
                if (metrics[m] >= cfg['standards'][m]
                    if cfg['higher_better'][m]
                    else metrics[m] <= cfg['standards'][m])
            )
            ratio = met / len(cfg['metrics'])
            rule_verdict = 'SELECT' if ratio==1.0 else 'WATCH' if ratio>=0.6 else 'PASS'

            # ML model
            ml_result = predict_player(metrics, pos, model_store)
            ml_verdict = ml_result['verdict']

            icon = {'SELECT':'✅','WATCH':'👀','PASS':'❌'}.get(rule_verdict,'❓')
            color = {'SELECT':'#f4c430','WATCH':'#e09c00','PASS':'#c0392b'}.get(rule_verdict,'gray')
            card_class = 'info-card' if rule_verdict != 'PASS' else 'red-card'

            st.markdown(
                f"<div class='{card_class}' style='text-align:center'>"
                f"<div style='font-size:3rem'>{icon}</div>"
                f"<h2 style='color:{color};margin:4px 0'>{rule_verdict}</h2>"
                f"<p style='color:#a8d5ba'>{met}/{len(cfg['metrics'])} standards met "
                f"({ratio*100:.0f}%)</p>"
                f"</div>",
                unsafe_allow_html=True)

            if ml_verdict and ml_verdict != '—':
                ml_icon  = {'SELECT':'✅','WATCH':'👀','PASS':'❌'}.get(ml_verdict,'❓')
                ml_color = {'SELECT':'#f4c430','WATCH':'#e09c00','PASS':'#c0392b'}.get(ml_verdict,'gray')
                st.markdown(
                    f"<div class='info-card' style='margin-top:8px'>"
                    f"<small style='color:#a8d5ba'>ML Model verdict:</small><br>"
                    f"<span style='color:{ml_color};font-weight:bold;font-size:1.1rem'>"
                    f"{ml_icon} {ml_verdict}</span>"
                    f"</div>",
                    unsafe_allow_html=True)

            advice = {
                'SELECT': '🌟 Player meets all analyst standards. Recommend immediate shortlist.',
                'WATCH':  '📋 Promising profile. Monitor closely — close to selection standard.',
                'PASS':   '📌 Below required threshold. Not yet at selection standard.',
            }.get(rule_verdict,'')
            st.info(advice)

            # Radar
            fake_row = pd.Series({**metrics, 'Label': rule_verdict,
                                  'Name': 'New Player', 'Club': '—'})
            radar_b  = make_radar(fake_row, pos)
            st.image(radar_b, use_container_width=True)
        else:
            st.markdown(
                "<div class='info-card' style='text-align:center;padding:40px'>"
                "<div style='font-size:2rem'>⚽</div>"
                "<p style='color:#52b788'>Enter metrics and click predict</p>"
                "</div>",
                unsafe_allow_html=True)


def page_model_results(model_store):
    st.markdown("## 🤖 Model Results")
    if not model_store:
        st.warning("No models trained yet. Upload a dataset to train.")
        return

    for pos, pkg in model_store.items():
        with st.expander(f"**{pos}**  — Best: `{pkg['model_name']}`  F1: `{pkg['f1']:.3f}`"):
            st.markdown("**Cross-Validation Results:**")
            cv_df = pd.DataFrame(list(pkg['cv_results'].items()),
                                  columns=['Model','F1 Score'])
            cv_df['Best'] = cv_df['Model'] == pkg['model_name']
            st.dataframe(cv_df.style.applymap(
                lambda v: 'color:#f4c430;font-weight:bold' if v == True else '',
                subset=['Best']), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Logo header ──────────────────────────────────────────────────────────
    logo_bytes = build_logo_svg(width=900, height=155)
    st.image(logo_bytes, use_container_width=True)
    st.markdown("<hr style='margin:0 0 12px 0;border-color:#c0392b;border-width:2px'>",
                unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    uploaded, page = render_sidebar()

    # ── Load data — uses session_state so CSV uploads never hit stale cache ──
    # Detect if a new file was uploaded this run
    uploaded_key = None
    if uploaded is not None:
        uploaded_key = f"{uploaded.name}_{uploaded.size}"

    if uploaded_key and st.session_state.get('loaded_key') != uploaded_key:
        # New file uploaded — clear cache and reprocess
        st.session_state.pop('pos_data', None)
        st.session_state.pop('model_store', None)
        st.session_state['loaded_key'] = uploaded_key

        file_bytes = uploaded.read()
        with st.spinner(f"Reading {uploaded.name}…"):
            df, err = load_data(file_bytes, uploaded.name)

        if err or df is None:
            st.error(f"❌ Failed to read file: {err}")
            return

        # Show column diagnostic so user can see what was detected
        with st.expander("🔍 Dataset Preview (expand to verify columns)", expanded=False):
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Columns found:** {list(df.columns)}")
            if 'Position' in df.columns:
                st.write(f"**Positions:** {df['Position'].value_counts().to_dict()}")
            st.dataframe(df.head(5))

        with st.spinner("Preprocessing data…"):
            pos_data = preprocess(df)

        if not pos_data:
            st.error(
                "❌ No valid player data found. "
                "Make sure your CSV has a **Position** column with values: "
                "Goalkeeper, Defender, Midfielder, Attacker — and the metric columns match.")
            st.write("**Your columns:**", list(df.columns))
            st.write("**Expected metric columns:**")
            for pos, cfg in POS_CONFIG.items():
                st.write(f"  {pos}: {cfg['metrics']}")
            return

        st.toast(f"✅ Loaded {sum(len(v) for v in pos_data.values())} players from {uploaded.name}", icon="⚽")

        with st.spinner("Training ML models (GradientBoosting, RandomForest, LogisticRegression)…"):
            model_store = train_models(pos_data)

        st.session_state['pos_data']    = pos_data
        st.session_state['model_store'] = model_store

    elif 'pos_data' in st.session_state:
        # Use previously loaded data from session
        pos_data    = st.session_state['pos_data']
        model_store = st.session_state.get('model_store', {})

    else:
        # Try local file — check csv first, then xlsx
        local = None
        for fname in ['i_league_players_dataset.csv',
                      'indian_football_dataset.csv',
                      'i_league_players_dataset.xlsx',
                      'indian_football_dataset.xlsx']:
            if os.path.exists(fname):
                local = fname
                break

        if local:
            with open(local, 'rb') as f:
                file_bytes = f.read()
            df, err = load_data(file_bytes, local)
            if err or df is None:
                st.error(f"❌ Failed to read local file: {err}")
                return
            pos_data = preprocess(df)
            if pos_data:
                with st.spinner("Training models…"):
                    model_store = train_models(pos_data)
                st.session_state['pos_data']    = pos_data
                st.session_state['model_store'] = model_store
            else:
                st.error("❌ Could not process local dataset.")
                return
        else:
            st.markdown(
                "<div class='red-card' style='text-align:center;padding:30px'>"
                "<h3 style='color:#e74c3c'>📂 No Dataset Found</h3>"
                "<p>Upload your <code>.csv</code> or <code>.xlsx</code> file using "
                "the <b>sidebar uploader</b>, or place <code>i_league_players_dataset.csv</code> "
                "in the same folder as this script.</p>"
                "</div>",
                unsafe_allow_html=True)
            return

    if not pos_data:
        st.error("❌ Dataset loaded but no players found. Check Position column values.")
        return

    # ── Route to page ─────────────────────────────────────────────────────────
    if page == "🏠 Overview":
        page_overview(pos_data)
    elif page == "👤 Players":
        page_players(pos_data)
    elif page == "📊 Analytics":
        page_analytics(pos_data)
    elif page == "🏆 Standings":
        page_standings(pos_data)
    elif page == "🗺 Tactics":
        page_tactics(pos_data)
    elif page == "🔮 Predict":
        page_predict(pos_data, model_store)

    # Model results always in footer expander
    if model_store:
        st.markdown("---")
        with st.expander("🤖 View Model Training Results"):
            page_model_results(model_store)


if __name__ == '__main__':
    main()

