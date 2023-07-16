
# birds-dist-models

This project provides a Streamlit application for running species distribution models. The user can select various parameters, including the model type, survey years, conservation ranks, species, and the variables.
The output is a map and a table of the species distribution.


## Usage
Clone the repository, install the dependencies, and in the project directory, run the following command in the terminal:
```
streamlit run src/app.py
```
If there are path issues you might need to run the following command first:

```
export PYTHONPATH="PROJECT_PATH"
```
where PROJECT_PATH is the path to the project directory.


## Data
At least 3 files are needed to run the model:
- survey_data.csv: the survey data file, in the following format:

|   | date       | latitude | longitude | survey_name      | species   | conservation_status | reserve_status       |
|---|------------|----------|-----------|------------------|-----------|---------------------|----------------------|
| 0 | 2018-02-16 | 30.15626 | 34.80496  | survey_name2018  | species A | LC                  | proposed_reservation |


- survey_features.csv: survey features file, where each row corresponds to an observation in the survey data file, and the columns are the features. 
The features can be categorical or numerical. For example:

|   | mean_annual_rainfall | mean_temp_jan | mean_temp_aug | max_temp_june | min_temp_jan |
|---|----------------------|---------------|---------------|---------------|--------------|
| 0 | 52.598541            | 10.856188     | 26.866346     | 39.453480     | 3.031685    |


- df_geo.csv or df_geo.geojson, a csv or geojson file for the predictions of the model outside the survey, in the following format:

|   | mean_annual_rainfall | mean_temp_jan | mean_temp_aug | max_temp_june | min_temp_jan | geometry |
|---|----------------------|---------------|---------------|---------------|--------------|----------|
| 0 | 36.517059            | 12.024206     | 29.077404     | 38.498764     | 3.663723    | POLYGON ((35.01329 29.92971, 35.01433 29.92971... |


where the geometry column contains a polygon in the format:

```
POLYGON ((35.01329194886916 29.92970855514185, 35.01432761819664 29.929710051883198, 35.0143259044598 29.930612145604048, 35.01329022579183 29.93061064880832, 35.01329194886916 29.92970855514185))
```

- Optional: reserve.shp file for plotting reservations on the probability map.


## Models
The following models are currently available:
- Logistic regression
- MaxEnt (using rpy2 to run the R package)
- Catboost





