# Digital Publishing Opportunity Intelligence

### Which Kindle niches should a new author write for?

**Author:** Stefan Radević · Data Science Bootcamp · Capstone project

> **Status:** I have finished collecting and reducing the data. My next step is Notebook 1 (EDA and feature
> engineering), followed by Notebook 2 (modelling). I will add the results here as I finish each stage.

---

## At a glance

| | |
|---|---|
| **Who it is for** | New and less-known authors who are deciding what kind of book to write next |
| **Main question** | Which Kindle niches are growing, how open are they to less-known authors, and how reliable is their audience? |
| **Prediction task** | I use what happens in a book's first 90 days to predict whether it reaches the top 25% of its niche within 12 months |
| **Second output** | A recommender that suggests niches and similar successful books to authors, and related books to readers |
| **Data** | One Kaggle snapshot (133,102 books) and the Amazon Reviews 2023 Kindle Store data (1.6M books, 25.3M ratings, 25.6M reviews) |
| **Time window** | 1 January 2019 to 13 September 2023, the last date in the data |
| **Tools** | Python, pandas, Polars, scikit-learn, Hugging Face Transformers, VADER |

---

## 1. The Data Science Question

**Main question**

> Which Kindle book niches show growing reader demand, how open are they to less-known authors,
> and can the signals a book shows in its first 90 days predict whether it will succeed within its niche?

**Sub-questions**

To answer the main question, I split it into five smaller ones:

1. **Demand:** I want to find out which niches gain a larger share of reader ratings from 2019 to 2023.
2. **Openness:** I check in which niches books by less-known authors reach the top of their niche, and how often this happens.
3. **Audience quality:** I look at where ratings and reviews are more reliable, using verified purchases, repeat raters
   and whether the review text agrees with the stars.
4. **Early prediction:** I test whether early signals (number of ratings, average rating, share of 5-star ratings,
   review sentiment, price and author fame) can predict success after 12 months, and whether this works equally well
   in every niche.
5. **Recommendation:** I want to suggest to an author the niches and successful books by less-known authors that are
   closest to what they want to write, and to suggest related books to readers.

**A hypothesis I test instead of assuming it.** I expect that in niches for children and learning, parents and students
leave reviews more often. To test this fairly, I compare reviews **within** each niche (using percentiles) rather than
across niches, and I check how many of the reviews come from verified purchases.

## 2. Why I Chose This Problem

Most advice for new authors is based on bestsellers. Bestsellers are mostly written by authors who already have a large
audience, so copying them tells a new author very little about where *they* can succeed. I wanted to look at the
market from the other side: where do books by **less-known** authors do well, and how early can we tell?

I designed the project as an **adviser for future authors**. Its main output is a ranked list of niches, together with
examples of successful books by less-known authors. I added a recommender for readers as an extra, built from the overlap
of people who rate books in different niches.

I did **not** throw the bestsellers away. I use them as a reference, to show what they look like, how they differ from
other books, and whether my definition of success agrees with Amazon's bestseller labels. I keep bestsellers by famous
authors separate from bestsellers by less-known authors, because the goal is not just to copy the famous ones.

---

## 3. Primary Datasets

### 3.1 Amazon Kindle Books Dataset 2023 (Kaggle)

| | |
|---|---|
| Source | [Kaggle: Amazon Kindle Books Dataset 2023 (130K Books)](https://www.kaggle.com/datasets/asaniczka/amazon-kindle-books-dataset-2023-130k-books) |
| File | `kindle_data-v2.csv` (37.5 MB) |
| Size | 133,102 books × 16 columns, 31 categories, one snapshot from 2023 |
| One row | one book |
| Key columns | `title`, `author`, `stars`, `reviews`, `price`, `isKindleUnlimited`, `isBestSeller`, `isEditorsPick`, `isGoodReadsChoice`, `publishedDate`, `category_name` |
| What I use it for | the bestseller reference, my earlier clustering and classification work, and the presentation |

**Problems I found**

- The `reviews` column is 0 for every book in 11 categories (including Children's eBooks), and more than half of the
  values are zeros in 15 other categories.
- `publishedDate` is missing for about 37% of the books, mostly in fiction.
- Each category seems to be capped at about 4,000–6,000 books, so the dataset does **not** show how big a market is.
- The books lean towards the top of Amazon's lists, so less-known authors are under-represented.

Because of these problems, this dataset alone cannot answer my main question. That is why I added the Amazon Reviews 2023 data.

### 3.2 Amazon Reviews 2023: Kindle Store (McAuley Lab, UC San Diego)

Source: [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/) (see the reference in section 9).
I use three files from the Kindle Store category:

| File | Size | Rows | Columns | One row | What I use it for |
|---|---|---:|---:|---|---|
| `meta_Kindle_Store.jsonl` | 6.9 GB | 1,591,371 | 16 | one book: title, author, categories, price, details | niches, author fame, book features |
| `Kindle_Store_0core.csv` | 1.5 GB | 25,302,305 | 4 | one rating: user, book, stars, time | early signals, demand per niche, recommender |
| `Kindle_Store_reviews.jsonl` | 15.8 GB | 25,577,616 | 10 | one review: stars, title, text, helpful votes, verified purchase | sentiment, audience reliability |

The ratings come from 5.6 million users and cover 1.59 million books, from August 1999 to **13 September 2023**.
A 5-core version also exists (only users and books with at least 5 ratings), but I chose **not** to use it. It keeps
the popular books and drops the "long tail" of less-known books, which is exactly the part this project is about.

**Problems I found while checking the files**

| Problem | How I handle it |
|---|---|
| The files are 1.5–15.8 GB | They cannot be opened in pandas, so I read them line by line or in chunks (section 4) |
| `author` is missing for 335,876 books (21%) | I recover the name from the `store` text, which usually says "Name (Author) Format: …" |
| `price` is sometimes a number and sometimes text (`"from 2.99"`), and it is missing for 215,559 books | I clean it in EDA, and I check why so many prices are `0.0` |
| `bought_together` is empty for all 1,591,371 books | I drop this column |
| `categories` almost always has only 3 levels (`Kindle Store > Kindle eBooks > Genre`) | A niche can only be a genre (about 31 real genres), so I get finer sub-niches from the titles |
| 63 different values at the genre level, and some of them are not genres (e.g. "Kindle Short Reads", promo lists) | I group or remove the non-genre values in EDA |
| Publication years such as 1800 and 2036 | I treat them as anomalies in EDA |
| There are 275,311 more reviews than ratings | I explain the difference in EDA (duplicates, edited reviews) |
| Some reviews are exact duplicates (same user, book, time and text) | I show them first and then remove them in EDA |
| The number of ratings per year falls after 2016 | This comes from how the dataset was collected, not from the market, so I measure demand as a **share** of all ratings |

---

## 4. From Raw Data to Working Data

The raw files are too large to open in a notebook, so I followed one simple rule: **scripts only make the data smaller,
the notebooks do the analysis.** My scripts choose rows and columns and keep every value exactly as it was. They do not
clean, join or create features, and they do not remove anomalies or duplicates. I do all of that in the notebooks, step by
step, so every decision can be seen and explained.

### 4.1 Decisions

| What I did | Why I did it |
|---|---|
| I kept ratings and reviews from **2019-01-01 to 2023-09-13** | This cuts the data by about two thirds, but still leaves enough years to see trends and a full 12 months of follow-up for books first rated before September 2022 |
| I kept the **full history** in two small count tables (first rating date, and ratings per book and year) | Without the full history, a book first rated in 2014 would look "new" in 2019, and an author who has been famous since 2015 would look "less known" |
| I kept **all 1.6M books** in the book table, but I do EDA on a **random sample of 200,000 books** rated in 2019–2023 | Author fame needs every book an author has written, while the sample keeps the notebook fast |
| I took a **balanced** review sample: up to 50,000 reviews from each of 4 niches, or all of them if a niche has fewer | A purely random sample was 55% Teen & Young Adult and only 4% Education & Teaching, so the small niches could not be compared |
| I kept **all reviews from a book's first 90 days**, but only for sampled books with 12 months of follow-up | This lets me link early reviews to success without looking into the future, and it keeps the file under 300 MB |
| I kept **random unchanged lines** from every raw file | This lets me show in the notebook what the original data looks like, column by column |

### 4.2 Scripts (`scripts/`)

| Script | What it reads | What it produces |
|---|---|---|
| `00_make_raw_samples.py` | all three raw files | random raw lines, completely unchanged |
| `01_make_meta_light.py` | `meta_Kindle_Store.jsonl` | a book table with the 13 useful columns, values unchanged |
| `02_make_ratings_2019_2023.py` | `Kindle_Store_0core.csv` | ratings from 2019–2023, the two count tables, and 200,000 sampled book IDs |
| `03_sample_reviews.py` | `Kindle_Store_reviews.jsonl` | the balanced review sample and the first-90-day reviews |

Each script runs in under 3 minutes on a laptop. I always ran each one on the first few hundred thousand lines first,
to check it worked before running it on the whole file. More details are in [`scripts/README.md`](scripts/README.md).

### 4.3 Working Datasets

**Raw samples** (`data/raw_sample/`). I use these to show what the original data looks like:

| File | Rows | Columns | Taken from |
|---|---:|---:|---|
| `meta_Kindle_Store_sample.jsonl` | 50,000 | 16 | random books |
| `Kindle_Store_reviews_sample.jsonl` | 100,000 | 10 | random reviews |
| `Kindle_Store_0core_sample.csv` | 100,000 | 4 | random ratings |

**Reduced data** (`data/clean/`). I use these for EDA, joins and feature engineering:

| File | Rows | Columns | Size | One row |
|---|---:|---:|---:|---|
| `meta_light.parquet` | 1,591,371 | 13 | 165 MB | one book (all books, all years) |
| `ratings_2019_2023.parquet` | 8,954,662 | 4 | 215 MB | one rating from 2019 on |
| `book_first_rating.parquet` | 1,590,697 | 3 | 25 MB | one book: time of its first rating and its ratings in all years |
| `book_year_counts.parquet` | 3,581,037 | 4 | 23 MB | one book in one year: number of ratings and sum of stars |
| `book_sample_ids.parquet` | 200,000 | 1 | 2 MB | the ID of one sampled book |
| `reviews_sample.parquet` | 141,872 | 9 | 42 MB | one review (4 niches, balanced) |
| `reviews_first_90_days.parquet` | 822,258 | 9 | 278 MB | one review from a sampled book's first 90 days |

All tables connect through `parent_asin`, the book ID. Ratings and reviews can also be connected through `user_id`.
These files do **not** contain the niche, author fame or success yet: I create those in Notebook 1.

---

## 5. Key Definitions

| Term | What I mean by it |
|---|---|
| **Niche** | The genre level of the Amazon category path, e.g. `Romance` or `Children's eBooks`. I get finer sub-niches from the titles |
| **Release date (proxy)** | I use the date of a book's **first rating**, because the publication date is often missing or wrong |
| **Success** (target) | A book is successful if it is in the **top 25% of its niche** by number of ratings in the **first 12 months** after its first rating |
| **Success cohort** | The books first rated between 2019-01-01 and 2022-09-12, so that every one of them has a full 12 months of follow-up (482,122 books; 108,847 of them are in my 200k sample) |
| **Early signals** | What is already known in the first 30–90 days: number of ratings, average rating, share of 5-star ratings and review sentiment |
| **Less-known author** | I decide this **per book**: I add up the ratings of the author's *other* books received *before* this book appeared, and if the total is below the median, the author counts as less known. The book itself is never counted |
| **Sentiment** | A 0/1 column for each review: 1 = positive, 0 = negative, predicted from the review **text** |
| **Bestseller label** | I use it only as an extra check of my definition of success, not as the target |

## 6. Guardrails

These are the rules I follow so that my results can be trusted:

- **No data leakage.** I take early signals only from the first 90 days, I measure success over 12 months, and I count
  author fame only from ratings received before the book appeared. This way the model never "sees the future".
- **Same source, separate windows.** Early signals and success both come from ratings, so I keep the two time windows
  strictly apart.
- **Within-niche comparison.** I compare ratings and reviews within a niche (using percentiles), because niches differ in
  size and in how often people write reviews.
- **Prediction, not causation.** My model can show that early signals *predict* success, but not that they *cause* it.
- **Reliability per niche.** I report the model's quality (ROC-AUC, PR-AUC and calibration) separately for each niche,
  together with how much the early signals add compared with the niche's base rate.

---

## 7. Suggestions from Mentors and How I Applied Them

Because my datasets are very large and because I need to understand what people write in their reviews, I was 
suggested a few tools that could help.

For reading reviews, **BERT** is suggested, a language model that reads a whole sentence at once instead of word by
word, together with the **Hugging Face Transformers** library, which makes such ready-made models easy to use, and
**PyTorch**, the library that actually runs the model. This is useful because BERT understands context: it knows that
"not that bad" is not a negative review, and it gives each review a score from 1 to 5 stars. Before this suggestion, I had
planned to use **VADER**, a simpler tool that only checks whether individual words are positive or negative. VADER is very
fast, but it is easily confused by phrases like "not bad".

For the large datasets, **Polars** is suggested, a library similar to pandas that can look through a very large file and
count what is inside without loading all of it into memory, and that can split big results into smaller Parquet files.
This is useful because my original files are up to 15.8 GB and cannot be opened at once.

What I use, where and why:

- **BERT (with Transformers and PyTorch)** in Notebook 1, during feature engineering, to mark every review as positive (1)
  or negative (0). I chose it as my main tool because it understands reviews better than a word list.
- **VADER** in Notebook 2, at the end, on a small sample of reviews, to compare the simple tool with BERT and see how much
  the smarter model really adds.
- **Polars** at the start of Notebook 1, to count the rows, columns and missing values in the full original files, so that
  everyone can see how much data there was at the beginning. I do not need it for splitting files, because I had already
  reduced the data with my own scripts and my largest working file is under 300 MB. For the rest of the analysis I use
  **pandas**, because the working files fit comfortably in memory.

---

## 8. Plan of Work

### Notebook 1: EDA and feature engineering (`notebooks/01_EDA_feature_engineering.ipynb`)

Every step in the notebook follows the same pattern: first I explain **what** I do and **why**, then I show the
**code**, and then I explain **what the result shows**.

1. **Primary datasets, from general to specific:** I start with the whole Kaggle file and the full Amazon files, counting
   their rows, columns and missing values, in order to show how much data there was at the beginning. Then I look closely
   at the raw samples and explain which parts I kept and why. Finally, I check that my reduced files match the counts in
   the raw files.
2. **Structure and quality of the working data:** I check the shapes, data types, missing values, duplicates and anomalies
   of each file, in order to know what needs to be fixed before any analysis.
3. **Cleaning:** I clean prices, recover missing authors from the `store` text, extract the niche, convert dates and remove
   duplicate reviews, in order to work only with correct and comparable values.
4. **Joins:** I join books with their ratings, reviews and author history, in order to have everything about a book in
   one table and to be able to compare books by niche and by author.
5. **Bestsellers:** I compare bestsellers with other books, and famous authors with less-known ones, in order to see what
   success looks like and how much of it comes from fame alone.
6. **EDA by niche:** I look at how demand changes over time, at the children's and education niches in detail, and at each
   niche's audience (unique and repeat raters, verified purchases, average rating), in order to find growing niches and
   to test my review hypothesis.
7. **Feature engineering:** I create the niche, author fame per book, the success target, the early signals, review
   sentiment (0/1) and price and title features, in order to give the models in Notebook 2 the information they need.
8. **Findings and tables for Notebook 2:** I summarise what I found and save the final tables, in order to start the
   modelling from clean, ready data.

### Notebook 2: Modelling (`notebooks/02_modeling.ipynb`)

- **Niche growth:** I forecast each niche's share of ratings, in order to see which niches are likely to keep growing.
- **Success prediction:** I build a classification model from the early signals and compare it with a simple baseline
  (the niche's base rate), in order to see whether early signals really help. I evaluate it separately for each niche
  with ROC-AUC, PR-AUC and calibration.
- **Ranking of niches for authors:** I combine growth, openness to less-known authors and audience reliability, in order
  to give authors one clear ranked list of niches.
- **Recommender:** I combine recommendations based on what similar readers rated with recommendations based on the books
  themselves. I sample whole **users** (with all their ratings) rather than random rows, split the data by time, and
  compare the results with simply recommending the most popular books (Recall@10, NDCG@10). I report popular books and
  less-known books separately, in order to make sure the recommender does not only suggest bestsellers.
- **Comparison of the two review tools:** I compare the simple and the smarter sentiment tool on a small sample, in order
  to see how much the smarter tool adds.

### Earlier work I keep as a reference

- I clustered the Kaggle data (K-Means with k = 3, hierarchical clustering and DBSCAN) and found three profiles: regular
  books (~52%), cheap Kindle Unlimited books (~27%) and books with long titles (~21%), with a silhouette score of about 0.3.
- I built a random forest model for the bestseller label, which reached a ROC-AUC of about 0.83.

---

## 9. Data Sources and References

- Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging Language and Items for Retrieval and
  Recommendation.* arXiv:2403.03952. Dataset: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
- Amazon Kindle Books Dataset 2023 (130K Books), Kaggle:
  [kaggle.com/datasets/asaniczka/amazon-kindle-books-dataset-2023-130k-books](https://www.kaggle.com/datasets/asaniczka/amazon-kindle-books-dataset-2023-130k-books).
- Sentiment model: [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) (Hugging Face).
- Hutto, C. J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* ICWSM.

I use the data for education only. The raw and processed data files are not included in this repository.
