import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import StringIO

st.set_page_config(
    page_title="OECD Public Spending Explorer",
    page_icon="📊",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────

OECD_COUNTRIES = {
    "AUS": "Australia", "AUT": "Austria", "BEL": "Belgium", "CAN": "Canada",
    "CHL": "Chile", "COL": "Colombia", "CRI": "Costa Rica", "CZE": "Czech Republic",
    "DNK": "Denmark", "EST": "Estonia", "FIN": "Finland", "FRA": "France",
    "DEU": "Germany", "GRC": "Greece", "HUN": "Hungary", "ISL": "Iceland",
    "IRL": "Ireland", "ISR": "Israel", "ITA": "Italy", "JPN": "Japan",
    "KOR": "Korea", "LVA": "Latvia", "LTU": "Lithuania", "LUX": "Luxembourg",
    "MEX": "Mexico", "NLD": "Netherlands", "NZL": "New Zealand", "NOR": "Norway",
    "POL": "Poland", "PRT": "Portugal", "SVK": "Slovak Republic", "SVN": "Slovenia",
    "ESP": "Spain", "SWE": "Sweden", "CHE": "Switzerland", "TUR": "Türkiye",
    "GBR": "United Kingdom", "USA": "United States",
}

COFOG_CATEGORIES = {
    "01": "General public services",
    "02": "Defence",
    "03": "Public order & safety",
    "04": "Economic affairs",
    "05": "Environmental protection",
    "06": "Housing & community amenities",
    "07": "Health",
    "08": "Recreation, culture & religion",
    "09": "Education",
    "10": "Social protection",
}

DEFAULT_COUNTRIES = ["USA", "GBR", "FRA", "DEU", "JPN"]
DEFAULT_CATEGORIES = {"07", "09", "10", "02"}  # Health, Education, Social protection, Defence

# ── Data Fetching ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(
    country_codes: tuple,
    category_codes: tuple,
    start_year: int,
    end_year: int,
):
    """
    Fetch government expenditure by COFOG function from the OECD SDMX API.
    Dataset: OECD.SDD.NAD, DSD_NASEC10@DF_TABLE11
    Dimensions: FREQ . REF_AREA . SECTOR . TRANSACT . ACTIVITY . UNIT_MEASURE
      - FREQ       : A  (annual)
      - REF_AREA   : country codes joined with +
      - SECTOR     : S13 (general government)
      - TRANSACT   : _T (total expenditure)
      - ACTIVITY   : COFOG codes joined with +
      - UNIT_MEASURE: XDC_R_B1GQ (national currency, ratio to GDP)
    """
    countries_str = "+".join(country_codes)
    categories_str = "+".join(category_codes)

    url = (
        "https://sdmx.oecd.org/public/rest/data/"
        f"OECD.SDD.NAD,DSD_NASEC10@DF_TABLE11,1.0/"
        f"A.{countries_str}.S13._T.{categories_str}.XDC_R_B1GQ"
        f"?startPeriod={start_year}&endPeriod={end_year}"
        "&dimensionAtObservation=AllDimensions&format=csvfilewithlabels"
    )

    try:
        resp = requests.get(url, timeout=45, headers={"Accept": "text/csv"})
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        return df, None, url
    except requests.exceptions.HTTPError as exc:
        return None, f"HTTP {exc.response.status_code}: {exc}", url
    except Exception as exc:
        return None, str(exc), url


def _find_col(columns: list[str], *candidates: str) -> str | None:
    """Case-insensitive column lookup."""
    upper = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in upper:
            return upper[cand.upper()]
    return None


def process_data(raw: pd.DataFrame) -> pd.DataFrame:
    cols = list(raw.columns)

    country_code_col  = _find_col(cols, "REF_AREA", "COUNTRY", "LOCATION")
    country_name_col  = _find_col(cols, "Reference area", "Country name", "COUNTRY_NAME")
    activity_code_col = _find_col(cols, "ACTIVITY", "FUNCTION")
    activity_name_col = _find_col(cols, "Economic activity", "Function name", "ACTIVITY_NAME")
    year_col          = _find_col(cols, "TIME_PERIOD", "TIME", "YEAR")
    value_col         = _find_col(cols, "OBS_VALUE", "Value", "VALUE")

    for name, col in [
        ("country code", country_code_col),
        ("activity code", activity_code_col),
        ("time period", year_col),
        ("observation value", value_col),
    ]:
        if col is None:
            raise ValueError(
                f"Could not find a column for '{name}'. "
                f"Available columns: {cols}"
            )

    df = pd.DataFrame(
        {
            "country_code":  raw[country_code_col].astype(str),
            "country_name":  raw[country_name_col].astype(str) if country_name_col else raw[country_code_col].astype(str),
            "activity_code": raw[activity_code_col].astype(str).str.zfill(2),
            "activity_name": raw[activity_name_col].astype(str) if activity_name_col else raw[activity_code_col].astype(str),
            "year":          pd.to_numeric(raw[year_col], errors="coerce"),
            "value":         pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )

    df["category"] = df["activity_code"].map(COFOG_CATEGORIES).fillna(df["activity_name"])
    df = df.dropna(subset=["year", "value"])
    df["year"] = df["year"].astype(int)
    return df


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    name_to_code = {v: k for k, v in OECD_COUNTRIES.items()}
    selected_names = st.multiselect(
        "Countries",
        options=sorted(OECD_COUNTRIES.values()),
        default=sorted(OECD_COUNTRIES[c] for c in DEFAULT_COUNTRIES),
    )
    selected_countries = tuple(name_to_code[n] for n in selected_names)

    st.subheader("Spending categories")
    selected_categories = []
    for code, label in COFOG_CATEGORIES.items():
        if st.checkbox(label, value=(code in DEFAULT_CATEGORIES), key=f"cat_{code}"):
            selected_categories.append(code)
    selected_categories = tuple(selected_categories)

    st.subheader("Year range")
    year_start, year_end = st.slider(
        "", min_value=2007, max_value=2023, value=(2010, 2022)
    )

# ── Validation ─────────────────────────────────────────────────────────────────

if not selected_countries:
    st.warning("Please select at least one country.")
    st.stop()
if not selected_categories:
    st.warning("Please select at least one spending category.")
    st.stop()

# ── Fetch & process ────────────────────────────────────────────────────────────

st.title("📊 OECD Public Spending Explorer")
st.caption("Government expenditure by COFOG function as % of GDP · Source: OECD")

with st.spinner("Fetching data from OECD…"):
    raw_df, error, api_url = fetch_data(
        selected_countries, selected_categories, year_start, year_end
    )

if error:
    st.error(f"Failed to fetch data: {error}")
    with st.expander("API URL"):
        st.code(api_url)
    st.stop()

try:
    df = process_data(raw_df)
except ValueError as exc:
    st.error(str(exc))
    with st.expander("Raw data (first 10 rows)"):
        st.dataframe(raw_df.head(10))
    st.stop()

df = df[df["year"].between(year_start, year_end)]

if df.empty:
    st.warning("No data returned for the selected filters. Try different countries or categories.")
    with st.expander("API URL"):
        st.code(api_url)
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_trends, tab_compare, tab_heatmap, tab_raw = st.tabs(
    ["📈 Trends over time", "📊 Country comparison", "🌡️ Heatmap", "🗂 Raw data"]
)

# ── Tab 1: Trends ──────────────────────────────────────────────────────────────

with tab_trends:
    st.subheader("Spending trends over time (% of GDP)")
    categories_present = sorted(df["category"].unique())

    cols = st.columns(min(2, len(categories_present)))
    for i, category in enumerate(categories_present):
        cat_df = df[df["category"] == category].sort_values("year")
        fig = px.line(
            cat_df,
            x="year",
            y="value",
            color="country_name",
            title=category,
            labels={"value": "% of GDP", "year": "Year", "country_name": "Country"},
            markers=True,
        )
        fig.update_layout(
            legend_title_text="Country",
            hovermode="x unified",
            title_font_size=14,
            margin=dict(t=40, b=20),
        )
        cols[i % 2].plotly_chart(fig, use_container_width=True)

# ── Tab 2: Country comparison bar chart ───────────────────────────────────────

with tab_compare:
    latest_year = df["year"].max()
    compare_year = st.slider(
        "Select year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=int(latest_year),
        key="compare_year",
    )
    year_df = df[df["year"] == compare_year]

    if year_df.empty:
        st.info(f"No data available for {compare_year}.")
    else:
        fig = px.bar(
            year_df.sort_values(["country_name", "category"]),
            x="country_name",
            y="value",
            color="category",
            barmode="group",
            title=f"Government expenditure by function — {compare_year} (% of GDP)",
            labels={"value": "% of GDP", "country_name": "Country", "category": "Category"},
        )
        fig.update_layout(xaxis_tickangle=-30, legend_title_text="Category")
        st.plotly_chart(fig, use_container_width=True)

        # Stacked bar for total view
        fig2 = px.bar(
            year_df.sort_values(["country_name", "category"]),
            x="country_name",
            y="value",
            color="category",
            barmode="stack",
            title=f"Total selected expenditure (stacked) — {compare_year} (% of GDP)",
            labels={"value": "% of GDP", "country_name": "Country", "category": "Category"},
        )
        fig2.update_layout(xaxis_tickangle=-30, legend_title_text="Category")
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Heatmap ─────────────────────────────────────────────────────────────

with tab_heatmap:
    hm_year = st.slider(
        "Select year",
        min_value=int(df["year"].min()),
        max_value=int(df["year"].max()),
        value=int(df["year"].max()),
        key="heatmap_year",
    )
    hm_df = df[df["year"] == hm_year]

    if hm_df.empty:
        st.info(f"No data available for {hm_year}.")
    else:
        pivot = hm_df.pivot_table(
            index="country_name", columns="category", values="value"
        )
        fig = px.imshow(
            pivot,
            text_auto=".1f",
            color_continuous_scale="Blues",
            title=f"Government expenditure heatmap — {hm_year} (% of GDP)",
            labels={"color": "% of GDP"},
            aspect="auto",
        )
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Raw data ────────────────────────────────────────────────────────────

with tab_raw:
    st.dataframe(
        df.sort_values(["country_name", "category", "year"]),
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False),
        file_name="oecd_spending.csv",
        mime="text/csv",
    )
    with st.expander("API URL used"):
        st.code(api_url)
