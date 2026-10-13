# Chicago Crime Count Prediction with Spark: Full Dataset vs Synopses

A PySpark project that predicts daily crime counts per police district and crime type, and measures what you gain and lose by training on a **sample (synopsis)** of the data instead of the **full dataset**.

Built for the *Big Data Processing and Analysis* course at the Technical University of Crete (School of Electrical and Computer Engineering). The full write-up is in [`Project_Report.pdf`](Project_Report_2020030137.pdf).

## What this project does

1. Loads the Chicago crimes dataset (8.29M rows, 22 columns, 1.8 GB, 2001 to 15/04/2025) into Spark.
2. Builds **synopses** (20%, 30% and 40% of the data) with a custom **distributed reservoir sampling** routine that works partition by partition, so the data never has to fit in one machine's memory.
3. Checks that the synopsis is faithful by running three aggregate queries on the full data and on a 20% synopsis and comparing results and run time.
4. Trains three Spark MLlib regression models on the full data and on each synopsis, and compares accuracy (R², RMSE, MAE) against training time.

## Method

**Sampling.** Each partition is sampled with reservoir sampling (first stage), then the combined sample is condensed to the target size (second stage). A 20% synopsis came out at 1,658,835 rows against a theoretical 1,658,842, a difference below 1%.

**Cleaning.** Missing `District` values are filled using the most frequent `District` for the record's `Beat`. Crime types with fewer than 4,000 records are grouped into a single `Rare Crime` category.

**Prediction task.** Records are grouped by `Primary Type`, `Year`, `Month`, `Day` and `District`, and the crime count of each group is the target. The target is log-transformed, `ln(count + 1)`, to reduce skew. `Primary Type` and `District` are string-indexed and one-hot encoded, and `Year`, `Month` and `Day` are used as features.

**Split.** Training uses 2001 to 2020. Testing uses 2021 to 2024. 2025 is excluded because it only covers January to April. All models, including those trained on synopses, are tested on the same test set.

**Models.** Gradient Boosted Trees, Decision Tree and Linear Regression (Spark MLlib).

## Results

### Is the synopsis representative?

Aggregate queries on a 20% synopsis matched the full dataset within 1% error and ran much faster:

| Query | Full dataset | 20% synopsis | Speedup |
| --- | --- | --- | --- |
| Days ranked by crime percentage | 58.5 s | 2.5 s | ~23x |
| Top 5 months by crime count | 47.7 s | 2.4 s | ~20x |
| Arrest percentage, domestic vs non-domestic | 85.3 s | 1.9 s | ~44x |

### Model accuracy and training time

| Training data | Model | R² | RMSE | MAE | Training time |
| --- | --- | --- | --- | --- | --- |
| Full | Gradient Boosted Trees | 0.745 | 0.466 | 0.357 | 537 s |
| Full | Decision Tree | 0.715 | 0.492 | 0.373 | 126 s |
| Full | Linear Regression | 0.550 | 0.618 | 0.478 | 157 s |
| 40% synopsis | Gradient Boosted Trees | 0.532 | 0.631 | 0.510 | 420 s |
| 40% synopsis | Decision Tree | 0.465 | 0.674 | 0.540 | 73 s |
| 40% synopsis | Linear Regression | 0.615 | 0.572 | 0.453 | 39 s |
| 30% synopsis | Gradient Boosted Trees | 0.418 | 0.703 | 0.576 | 363 s |
| 30% synopsis | Decision Tree | 0.334 | 0.752 | 0.613 | 66 s |
| 30% synopsis | Linear Regression | 0.631 | 0.560 | 0.456 | 31 s |
| 20% synopsis | Gradient Boosted Trees | 0.139 | 0.855 | 0.703 | 334 s |
| 20% synopsis | Decision Tree | 0.067 | 0.890 | 0.731 | 59 s |
| 20% synopsis | Linear Regression | 0.475 | 0.668 | 0.550 | 25 s |

### Takeaways

- Training on the full data with Gradient Boosted Trees gives the best accuracy (R² 0.745), at the highest training cost.
- The Decision Tree trains about 4.3x faster than Gradient Boosted Trees with only a small accuracy loss.
- The tree-based models degrade steadily as the synopsis shrinks, so they depend strongly on data volume.
- Linear Regression is robust to smaller samples. On the 30% synopsis it reaches R² 0.631 and trains about 17x faster than Gradient Boosted Trees on the full data.

Timings come from single runs on Google Colab, so treat them as indicative rather than exact.

## How to run

The notebook was developed in **Google Colab**.

1. **Get the data.** Download *Crimes - 2001 to Present* from the [Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) as CSV. The dataset is about 1.8 GB, so it is **not** included in this repository. The notebook was run on the export dated 15/04/2025 (`Crimes_-_2001_to_Present_20250415.csv`).
2. **Upload the CSV to Google Drive** and set the `file_path` variable in the notebook to its location.
3. **Open** `chicago-crime-spark-synopses.ipynb` in Google Colab and run the cells in order. The first cells install `pyspark`, `findspark` and `pyngrok`.
4. **Optional: Spark UI.** The setup cell can expose the Spark web UI through an ngrok tunnel. This needs your own free ngrok authtoken, which the notebook asks for at runtime. Never paste the token into the notebook itself. You can skip this step and everything else still works.

## Repository contents

| File | Description |
| --- | --- |
| `chicago-crime-spark-synopses.ipynb` | The full notebook: setup, sampling, queries, training and evaluation |
| `Project_Report.pdf` | The course report with methodology and detailed results |

## Possible improvements

- Use stratified sampling so that rare crime types and small districts are better represented in the synopses.
- Test more synopsis sizes to map the speed and accuracy trade-off more finely.
- Try neural networks as an alternative to classic regression models.

## Author

Koutsovasilis Vasileios | Electrical and Computer Engineering, Technical University of Crete
