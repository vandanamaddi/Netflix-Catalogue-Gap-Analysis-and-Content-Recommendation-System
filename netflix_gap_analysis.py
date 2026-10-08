"""
Foundations of Data Science Project
Netflix Catalogue Gap Analysis and Content Recommendation System
-----------------------------------------------------------------
Dataset : netflix_titles.csv
Tools   : Python, Pandas, NumPy, Matplotlib, Seaborn
Idea    : Count-based EDA to find under-represented areas (genres, countries,
          content type, years) and a rule-based recommendation table.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # remove this line if you run in Jupyter / IDE
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

DATA_PATH = "netflix_titles.csv"      # change path if needed
OUT_DIR = "graphs"
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(name):
    """Tidy layout, save the current figure and close it."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150)
    plt.close()


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ======================================================================
# 1. LOAD AND UNDERSTAND THE DATASET
# ======================================================================
heading("1. LOAD AND UNDERSTAND THE DATASET")

df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())
print("\nLast 5 rows:")
print(df.tail())
print("\nShape (rows, columns):", df.shape)
print("\nColumn names:", list(df.columns))
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nDuplicate records:", df.duplicated().sum())
print("Duplicate show_id values:", df["show_id"].duplicated().sum())
print("\nBasic statistics (numeric):")
print(df.describe())
print("\nBasic statistics (text columns):")
print(df.describe(include="object"))

# ======================================================================
# 2. DATA CLEANING
# ======================================================================
heading("2. DATA CLEANING")

# 2.1 Remove duplicates (rows that are exact copies)
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicates removed: {before - len(df)}  (rows now: {len(df)})")

# 2.2 Fix misplaced values: in 3 rows the duration (e.g. '74 min') was stored
#     in the 'rating' column and 'duration' is empty. Move it back.
misplaced = df["rating"].str.contains("min", na=False)
print("Rows with duration wrongly stored in rating:", misplaced.sum())
df.loc[misplaced, "duration"] = df.loc[misplaced, "rating"]
df.loc[misplaced, "rating"] = np.nan

# 2.3 Handle missing values (we do NOT drop rows; we fill with labels)
df["director"] = df["director"].fillna("Unknown")
df["cast"] = df["cast"].fillna("Unknown")
df["country"] = df["country"].fillna("Unknown")
df["rating"] = df["rating"].fillna("Not Rated")

# 2.4 Clean text spacing, then convert date_added to datetime
df["date_added"] = df["date_added"].str.strip()
df["date_added"] = pd.to_datetime(df["date_added"], format="%B %d, %Y", errors="coerce")
# date_added stays NaT for ~10 rows: filling with a made-up date would be
# wrong, so we leave them empty and just skip them in year/month analysis.

# 2.5 Duration: only 3 missing and they were fixed above; any left stay NaN
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# ======================================================================
# 3. FEATURE CREATION
# ======================================================================
heading("3. FEATURE CREATION")

df["year_added"] = df["date_added"].dt.year
df["month_added"] = df["date_added"].dt.month

# duration looks like '90 min' (Movie) or '2 Seasons' (TV Show).
# Take the number part, then put it in the correct column by type.
df["duration_value"] = pd.to_numeric(df["duration"].str.extract(r"(\d+)")[0])
df["movie_minutes"] = np.where(df["type"] == "Movie", df["duration_value"], np.nan)
df["tv_seasons"] = np.where(df["type"] == "TV Show", df["duration_value"], np.nan)

# number of genres in 'listed_in'
df["num_genres"] = df["listed_in"].str.split(",").apply(len)

print(df[["type", "duration", "duration_value", "movie_minutes", "tv_seasons",
          "date_added", "year_added", "month_added", "num_genres"]].head(8))

# ----------------------------------------------------------------------
# Helper tables: split multi-value columns (genres / countries) and count
# ----------------------------------------------------------------------
def split_count(column, drop_unknown=False):
    """Split a comma separated column, count each value separately."""
    items = df[column].str.split(",").explode().str.strip()
    items = items[items != ""]
    if drop_unknown:
        items = items[items != "Unknown"]
    return items.value_counts()


genre_counts = split_count("listed_in")
country_counts = split_count("country", drop_unknown=True)   # 'Unknown' is not a real country

# ======================================================================
# 4. EXPLORATORY DATA ANALYSIS
# ======================================================================
heading("4. EXPLORATORY DATA ANALYSIS")

type_counts = df["type"].value_counts()
print("4.1 Movies vs TV Shows:")
print(type_counts)
print((type_counts / len(df) * 100).round(1).astype(str) + " %")

print("\n4.2 Top 10 genres:")
print(genre_counts.head(10))

print("\n4.3 Top 10 countries:")
print(country_counts.head(10))

added_by_year = df["year_added"].dropna().astype(int).value_counts().sort_index()
print("\n4.4 Content added by year:")
print(added_by_year)

released_by_year = df["release_year"].value_counts().sort_index()
print("\n4.5 Content released by year (last 10 years):")
print(released_by_year.tail(10))

rating_counts = df["rating"].value_counts()
print("\n4.6 Ratings:")
print(rating_counts)

print("\n4.7 Movie duration (minutes):")
print(df["movie_minutes"].describe().round(1))

print("\n4.8 TV Show seasons:")
print(df["tv_seasons"].value_counts().sort_index())

# ======================================================================
# 5. CATALOGUE GAP ANALYSIS (frequency based)
# ======================================================================
heading("5. CATALOGUE GAP ANALYSIS")

# 5.1 Least represented genres
print("5.1 Least represented genres (bottom 10):")
print(genre_counts.tail(10))

# 5.2 Least represented countries
# Many countries have just 1 title, so we first look at the whole picture.
print("\n5.2 Countries with only 1 title :", (country_counts == 1).sum())
print("    Countries with fewer than 5 titles :", (country_counts < 5).sum())
print("    Total countries :", len(country_counts))
print("\nBottom 10 countries:")
print(country_counts.tail(10))

# 5.3 Movies vs TV Shows difference
movies, shows = type_counts["Movie"], type_counts["TV Show"]
print(f"\n5.3 Movies = {movies}, TV Shows = {shows}, "
      f"difference = {movies - shows}, ratio = {movies / shows:.2f} movies per TV show")

# Movie vs TV Show split inside each of the Top 10 countries
top10_countries = country_counts.head(10).index
rows = df.assign(country=df["country"].str.split(",")).explode("country")
rows["country"] = rows["country"].str.strip()
type_by_country = rows.groupby(["country", "type"]).size().unstack(fill_value=0).loc[top10_countries]
type_by_country["TV share %"] = (type_by_country["TV Show"] /
                                 type_by_country.sum(axis=1) * 100).round(1)
print("\nMovie / TV Show split in top 10 countries:")
print(type_by_country)

# 5.4 Years with low additions
# 2021 is a partial year in this dataset, so it is NOT treated as low.
full_years = added_by_year.drop(index=2021, errors="ignore")
low_year_limit = full_years.mean() * 0.25            # rule: below 25% of the average
low_years = full_years[full_years < low_year_limit]
print(f"\n5.4 Average additions per full year = {full_years.mean():.0f}; "
      f"low-year limit (25% of average) = {low_year_limit:.0f}")
print("Years with low additions:")
print(low_years)

# Release-year gap: old titles (before 2000) as a share of the catalogue
old_titles = (df["release_year"] < 2000).sum()
print(f"\nTitles released before 2000: {old_titles} "
      f"({old_titles / len(df) * 100:.1f}% of catalogue)")

# ----------------------------------------------------------------------
# 5.5 Gap level rule (simple, based on count percentiles)
# ----------------------------------------------------------------------
def gap_level(count, counts_series):
    """High / Medium / Low gap depending on where the count falls."""
    p10, p25 = counts_series.quantile(0.10), counts_series.quantile(0.25)
    if count <= p10:
        return "High Gap"
    elif count <= p25:
        return "Medium Gap"
    return "Low Gap"


# ----------------------------------------------------------------------
# 5.6 TOP 10 POTENTIAL CATALOGUE GAPS
# Selection rule (easy to explain):
#   - 4 least represented GENRES (lowest counts)
#   - 3 least represented COUNTRIES (lowest counts, ties broken alphabetically)
#   - 1 content type imbalance (TV Shows vs Movies)
#   - 2 low-addition years
# ----------------------------------------------------------------------
# 'Movies' and 'TV Shows' are generic catch-all tags, not real genres -> skip them
real_genres = genre_counts.drop(["Movies", "TV Shows"], errors="ignore")
gap_genres = real_genres.sort_values(ascending=True, kind="stable").head(4)

# 38 countries have only 1 title, so many counts are tied. Rule: take the countries
# with the lowest counts; ties are broken alphabetically (sort_index first).
one_title_countries = sorted(country_counts[country_counts == 1].index)
print("Countries with exactly 1 title:", ", ".join(one_title_countries))
gap_countries = country_counts.sort_index().sort_values(kind="stable").head(3)

gap_rows = []
for g, c in gap_genres.items():
    gap_rows.append(("Genre: " + g, c, gap_level(c, genre_counts)))
for ctry, c in gap_countries.items():
    gap_rows.append(("Country: " + ctry, c, gap_level(c, country_counts)))

# content-type imbalance
share_tv = shows / (movies + shows) * 100
tv_level = "High Gap" if share_tv < 35 else ("Medium Gap" if share_tv < 45 else "Low Gap")
gap_rows.append(("Content type: TV Shows", shows, tv_level))

# two lowest full years
for yr, c in low_years.sort_values().head(2).items():
    gap_rows.append((f"Year added: {yr}", c, "High Gap"))

gap_table = pd.DataFrame(gap_rows, columns=["Category", "Count", "Gap Level"])

# ======================================================================
# 6. VISUALIZATIONS
# ======================================================================
heading("6. VISUALIZATIONS (saved in the 'graphs' folder)")

# 6.1 Movies vs TV Shows
plt.figure(figsize=(6, 5))
ax = sns.barplot(x=type_counts.index, y=type_counts.values,
                 hue=type_counts.index, palette=["#E50914", "#564d4d"], legend=False)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.title("Movies vs TV Shows on Netflix")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")
save_fig("01_movies_vs_tvshows.png")

# 6.2 Top 10 genres
plt.figure(figsize=(9, 5))
sns.barplot(x=genre_counts.head(10).values, y=genre_counts.head(10).index,
            hue=genre_counts.head(10).index, palette="Reds_r", legend=False)
plt.title("Top 10 Genres on Netflix")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")
save_fig("02_top10_genres.png")

# 6.3 Top 10 countries
plt.figure(figsize=(9, 5))
sns.barplot(x=country_counts.head(10).values, y=country_counts.head(10).index,
            hue=country_counts.head(10).index, palette="Blues_r", legend=False)
plt.title("Top 10 Countries by Number of Titles")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
save_fig("03_top10_countries.png")

# 6.4 Content added by year
plt.figure(figsize=(10, 5))
plt.plot(added_by_year.index, added_by_year.values, marker="o", color="#E50914")
plt.title("Content Added to Netflix by Year (2021 is a partial year)")
plt.xlabel("Year Added")
plt.ylabel("Number of Titles")
save_fig("04_content_added_by_year.png")

# 6.5 Content released by year (from 1980 so the graph is readable)
recent = released_by_year[released_by_year.index >= 1980]
plt.figure(figsize=(10, 5))
plt.plot(recent.index, recent.values, marker="o", color="#564d4d")
plt.title("Content Released by Year (1980 onwards)")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
save_fig("05_content_released_by_year.png")

# 6.6 Top ratings
top_ratings = rating_counts.head(10)
plt.figure(figsize=(9, 5))
sns.barplot(x=top_ratings.index, y=top_ratings.values,
            hue=top_ratings.index, palette="viridis", legend=False)
plt.title("Top 10 Content Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")
plt.xticks(rotation=45)
save_fig("06_top_ratings.png")

# 6.7 Movie duration distribution
plt.figure(figsize=(9, 5))
sns.histplot(df["movie_minutes"].dropna(), bins=30, kde=True, color="#E50914")
plt.title("Distribution of Movie Duration")
plt.xlabel("Duration (minutes)")
plt.ylabel("Number of Movies")
save_fig("07_movie_duration_distribution.png")

# 6.8 TV show seasons (extra)
season_counts = df["tv_seasons"].value_counts().sort_index()
plt.figure(figsize=(9, 5))
sns.barplot(x=season_counts.index.astype(int), y=season_counts.values,
            hue=season_counts.index.astype(int), palette="Greys_r", legend=False)
plt.title("TV Shows by Number of Seasons")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of TV Shows")
save_fig("08_tvshow_seasons.png")

# 6.9 Gap chart: Top 10 gaps (extra)
plt.figure(figsize=(9, 5))
sns.barplot(x=gap_table["Count"], y=gap_table["Category"],
            hue=gap_table["Category"], palette="OrRd_r", legend=False)
plt.title("Top 10 Potential Catalogue Gaps (lower count = bigger gap)")
plt.xlabel("Number of Titles")
plt.ylabel("")
save_fig("09_top10_catalogue_gaps.png")

print("9 graphs saved.")

# ======================================================================
# 7. RULE-BASED RECOMMENDATION SYSTEM
# ======================================================================
heading("7. RULE-BASED RECOMMENDATION SYSTEM")


def recommend(category, count, level):
    """Return a simple text recommendation based on the type of gap."""
    if category.startswith("Genre:"):
        return "Netflix could consider adding more content in this genre."
    if category.startswith("Country:"):
        return ("This country is underrepresented and could provide new "
                "content opportunities.")
    if category.startswith("Content type"):
        return ("TV Shows are less represented than Movies, so additional TV "
                "content could be considered.")
    if category.startswith("Year added"):
        return ("Few titles were added in this year; older catalogue from this "
                "period could be acquired or licensed.")
    return "No action needed."


gap_table["Recommendation"] = [recommend(c, n, l) for c, n, l in
                               zip(gap_table["Category"], gap_table["Count"],
                                   gap_table["Gap Level"])]
gap_table.index = range(1, len(gap_table) + 1)
print(gap_table.to_string())

gap_table.to_csv("catalogue_gap_recommendations.csv", index_label="Rank")
df.to_csv("netflix_cleaned.csv", index=False)

# ======================================================================
# 8. FINAL OUTPUT
# ======================================================================
heading("8. FINAL OUTPUT")

print("MAJOR CATALOGUE GAPS")
print("- Top underrepresented genres   :", ", ".join(gap_genres.index))
print("- Top underrepresented countries:", ", ".join(gap_countries.index))
print(f"- Content-type imbalance        : {movies} Movies vs {shows} TV Shows "
      f"(TV Shows = {share_tv:.1f}% of catalogue)")
print("- Important temporal gaps       : very few titles added in",
      ", ".join(str(y) for y in low_years.index),
      f"; only {old_titles / len(df) * 100:.1f}% of titles released before 2000")

print("\nRECOMMENDATIONS")
print("- Content to add : more TV series, plus titles in the low-count genres listed above.")
print("- Countries      : explore the underrepresented countries listed above for fresh content.")
print("- Genres         : expand the least represented genres and keep strong genres balanced.")

print("\nProject finished. Outputs: graphs/, catalogue_gap_recommendations.csv, netflix_cleaned.csv")
