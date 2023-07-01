import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from plotnine import *
import matplotlib.patches as mpatches

import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import streamlit as st
from src.models import *
from src.utils import *
from src.config import *

from sklearn.model_selection import train_test_split

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import log_loss

from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from jinja2 import Template
from shapely import wkt

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


#DEBUG = True
DEBUG = False

SEED = 5

@st.cache_data
def load_data():
    to_read_file = True

    if to_read_file:
        df_ar_sub = gpd.read_file('data/df_ar_sub_ndvi.geojson')

    else:
        path_arava_data = "data/dbo.arava_polygon_w_biogis_layers.xlsx"
        df_arava_orig = pd.read_excel(path_arava_data)
        df_arava = df_arava_orig.copy()
        df_arava['geometry'] = df_arava['cell_coords'].apply(wkt.loads)
        df_arava = df_arava.drop(columns=['cell_coords'])
        df_arava = gpd.GeoDataFrame(df_arava, geometry='geometry')
        df_arava.crs = {'init' :'epsg:2039'}
        df_arava = df_arava.to_crs(epsg=4326)
        df_arava = df_arava.rename(columns=vars_gis_new_names)
        df_arava = df_arava.dropna()
        df_ar_sub = df_arava.sample(50000)
        df_ar_sub.to_file("data/df_ar_sub.geojson")
    df_ar = df_ar_sub.copy()
    df_ar['human_impact'] = df_ar['human_impact'].fillna(0)
    df_ar = df_ar.dropna()

    df_orig = pd.read_csv(data_wt_gis_path)
    df_birds,  df_cls = preproc(df_orig, to_impute=True, df_ar=df_ar)

    shm_negev = gpd.read_file('data/shm_negev.shp')

    return df_birds, df_cls, df_ar, shm_negev


def read_negev():
    # unzip the file
    # with zipfile.ZipFile("GIS_data/Nafot.zip","r") as zip_ref:
    #     zip_ref.extractall("GIS_data/")
    
    df_negev = gpd.read_file("GIS_data/נפות ישראל.shp")
    df_negev = df_negev[df_negev['Nafa'] == "באר שבע"]
    df_negev = df_negev.to_crs(epsg=4326)
    return df_negev



def preproc(df_survey, to_impute = True, df_ar=None):

    cols_to_select = vars_survey + vars_gis + vars_ndvi
    df_survey['date'] = pd.to_datetime(df_survey['date'])
    df_survey['month'] = df_survey['date'].dt.month
    df_survey['year'] = df_survey['date'].dt.year

    convs_map = pd.DataFrame({'status_name': ['LC', 'NT', 'VU', 'EN', 'CR'], 'status_rank': [1, 2, 3, 4, 5]})
    df_survey['conservation_rank'] = df_survey['conservation_status'].map(convs_map.set_index('status_name')['status_rank'])
     
    convs_map_2 = pd.DataFrame({'status_name': ['LC', 'NT', 'VU', 'EN', 'CR'], 
                                'status_full_name': ['Least Concerned', 'Near Threatened', 'Vulnerable', 'Endangered', 'Critically Endangered']})

    df_survey['conservation_status_full'] = df_survey['conservation_status'].map(convs_map_2.set_index('status_name')['status_full_name'])
    #df['threatened'] = df['conservation_status'].isin(['VU', 'EN', 'CR']).astype(int).astype('category')

    df_survey['year'] = df_survey['year'].astype('category')

    df_survey = df_survey.rename(columns={'latitude': 'y', 'longitude': 'x'})

    if "ndvi" in df_survey.columns:
        df_survey['ndvi'] = df_survey['ndvi'].astype(float)
        df_survey['ndvi'] = df_survey.groupby(['x', 'y'])['ndvi'].transform('mean')

    df_survey['bird_observed'] = ~df_survey['species'].isna()

    df_survey = gpd.GeoDataFrame(df_survey, geometry=gpd.points_from_xy(df_survey.x, df_survey.y))

    if to_impute:
        df_survey = impute_using_nearest_neighbor(df_survey, df_ar, vars_gis_cont + vars_gis_cat)

    df_survey = df_survey.copy()
    df_survey = df_survey.reset_index(drop=True)

    df_spc_obs_only = df_survey[~df_survey.species.isna()]
    df_spc_obs_only = df_spc_obs_only.reset_index(drop=True)

    return df_spc_obs_only, df_survey


def impute_using_nearest_neighbor(df_survey, df_all, vars_to_impute):
    df_all['centroid'] = df_all['geometry'].apply(lambda x: x.centroid)
    df_all['x'] = df_all['centroid'].apply(lambda x: x.x)
    df_all['y'] = df_all['centroid'].apply(lambda x: x.y)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(df_all[['x', 'y']])

    for var in vars_to_impute:
        if df_survey[var].isnull().any():
            missing_rows = df_survey[var].isnull()
            nearest_indices = nn.kneighbors(df_survey.loc[missing_rows, ['x', 'y']], return_distance=False).flatten()
            df_survey.loc[missing_rows, var] = df_all.iloc[nearest_indices][var].values
    return df_survey


def make_single_bird_labels(df_survey, spc=None):
    df_survey = df_survey.copy()

    labels = np.where(df_survey['species'].isin(spc), 1, 0).reshape(-1, 1)
    df_survey['label'] = labels

    df_survey = (
        df_survey
        .sort_values(by=['x', 'y', 'year', 'label'], ascending=False)
        .drop_duplicates(subset=['x', 'y', 'year'], keep='first')
        .reset_index(drop=True)
        )
    
    labels = df_survey['label'].values.reshape(-1, 1)
    return df_survey, labels

def preproc_for_model(df_cls, df_arava, cfg):

    df_cls = df_cls.copy()

    survey_years = cfg['survey_years']
    df_cls = df_cls.query('year in @survey_years').reset_index(drop=True)

    spc = cfg['species']

    assert set(spc).issubset(set(df_cls['species'].unique()))

    df_cls, y_train = make_single_bird_labels(df_cls, spc=spc)

    features = cfg['features']
    X_train = df_cls.loc[:, features]

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')

    if df_arava is not None:
        df_arava = df_arava.copy().reset_index(drop=True)

        if 'year' in df_arava.columns:
            df_arava = df_arava.query('year in @survey_years').reset_index(drop=True)

        df_arava = df_arava.loc[:, features]
        for col in df_arava.columns:
            if df_arava[col].dtype == 'object':
                df_arava[col] = df_arava[col].astype('category')
    
    return X_train, y_train, df_arava

def run_exp(model, 
            df_cls, 
            df_arava, 
            cfg=None,
            with_arava_preds=True):

    X_train, y_train, df_arava = preproc_for_model(df_cls, df_arava, cfg)

    model = model.fit(X_train, y_train)
    y_pred_train = model.predict_proba(X_train)

    res = dict()
    features = cfg['features']

    if with_arava_preds:
        X_arv = df_arava[features]
        y_pred_arv = model.predict_proba(X_arv)
        res['y_pred_arv'] = y_pred_arv

    res['y_pred_train'] = y_pred_train
    res['model'] = model
    res['y_train'] = y_train
    return res

def run_exp_for_each_species(model, df_cls, df_ar, cfg):
    res = {}
    for spc in cfg['species']:
        spc_cfg = cfg.copy()
        spc_cfg['species'] = [spc]
        spc_res = run_exp(model, df_cls, df_ar, cfg=spc_cfg)
        res[spc] = spc_res
    return res


def plot_probas_on_map(df_ar, 
                       df_birds, 
                       probas_list, 
                       spc_list=None, 
                       figsize=(10, 10), 
                       resolution=100, 
                       aggregation_type="Group together",
                       plot_other_species=False,
                       plot_nature_reserves=False,
                       shm_negev=None,
                       ):
    df_birds = gpd.GeoDataFrame(df_birds, geometry=gpd.points_from_xy(df_birds.x, df_birds.y))

    if aggregation_type == "Treat separately":
        probas = np.mean(probas_list, axis=0)

    else:
        probas = probas_list[0]

    df_ar_probas = df_ar.copy()
    df_ar_probas['pred_proba'] = probas

    # Calculate centroids and create interpolation grid
    df_ar_probas['centroid'] = df_ar_probas.geometry.centroid
    x = np.array([pt.x for pt in df_ar_probas.centroid])
    y = np.array([pt.y for pt in df_ar_probas.centroid])
    z = df_ar_probas['pred_proba'].values
    xi, yi = np.mgrid[min(x):max(x):resolution*1j, min(y):max(y):resolution*1j]

    # Perform IDW interpolation
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    # Plot interpolated data
    fig, ax = plt.subplots(figsize=figsize)
    cs = ax.contourf(xi, yi, zi, cmap='viridis', levels=10)
    fig.colorbar(cs, ax=ax)

    if spc_list:
        # only with the species
        df_birds_spc = df_birds.query('species in @spc_list')
        p = df_birds_spc.plot(ax=ax, marker='o', color='red', markersize=6,
                              label="chosen species")
        if plot_other_species:
            df_birds_other = df_birds.query('species not in @spc_list')
            # and also not the same coordinates as the chosen species
            df_birds_other = df_birds_other.query('x not in @df_birds_spc.x and y not in @df_birds_spc.y')
            df_birds_other.plot(ax=ax, marker='o', color='white', markersize=0.5,
                                alpha=0.5, label='other species')

        # set legend
        ax.legend()

    else:
        # plot all survey points
        df_birds.plot(ax=ax, marker='o', color='pink', markersize=3)

    if plot_nature_reserves and shm_negev is not None:

        colors = ['brown', 'hotpink', 'white', 'coral']  # replace with the actual colors
        cmap = ListedColormap(colors)

        status_to_color = {status: color for status, color in zip(shm_negev['STATUS_DES'].unique(), colors)}

        # Create a new column in the DataFrame mapping each 'STATUS_DES' value to its corresponding color
        shm_negev['color'] = shm_negev['STATUS_DES'].map(status_to_color)
        shm_negev['color'] = shm_negev['color'].fillna('black')

        shm_negev.boundary.plot(ax=ax, edgecolor=shm_negev['color'], linewidth=2.0,
                                linestyle='--', alpha=0.6)

        shm_negev_fill = shm_negev.copy()
        shm_negev_fill['STATUS_DES'] = 'a'

        shm_negev_fill.plot(ax=ax,
                       edgecolor=shm_negev_fill['color'],
                       column='STATUS_DES',
                       alpha=0.03,
                       color="white"
                       )

        color_mapping = dict(zip(shm_negev['STATUS_DES'].unique(), shm_negev['color'].unique()))
        patches = [mpatches.Patch(color=color, label=label) for label, color in color_mapping.items()]

        ax.legend(handles=patches, bbox_to_anchor=(-0.5, 1), loc='upper left', borderaxespad=0.)

        xmin, ymin, xmax, ymax = df_ar.total_bounds
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    # set title
    title = "Probability for bird presence"
    ax.set_title(title)

    return fig, ax


def plot_dot_whisker(coefs_stats, figsize=(5,6), dpi=100):
    coefs = pd.DataFrame(coefs_stats.params, columns=['coef'])
    coefs['std'] = coefs_stats.bse
    coefs['pval'] = coefs_stats.pvalues
    coefs['abs_coef'] = coefs.coef.abs()
    coefs.sort_values(by='abs_coef', ascending=False)

    to_include = coefs[1:].sort_values(by='abs_coef') 
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(to_include.coef, range(len(to_include)), color="#1a9988", zorder=2)
    ax.set_yticks(range(len(to_include)), to_include.index) # label the y axis with the ind. variable names
    ax.set_xlabel("Coefficient Value")

    for idx, ci in enumerate(coefs_stats.conf_int().loc[to_include.index].iterrows()):
        ax.hlines(idx, ci[1][0], ci[1][1], color="#eb5600", zorder=1, linewidth=3)

    p = plt.axline((0,0), (0,1), color="#eb5600", linestyle="--") # add a dashed line at 0.0
    title = "Coefficients and 95% Confidence Intervals"
    ax.set_title(title)

    #return fig, ax
    return fig

def plot_feature_on_map(df_ar, df_birds, variable, spc=None, plot_all_survey_points=False, 
                        buffer_distance=0.01, 
                        resolution=100,
                        figsize=(10,10), 
                        #dpi=100,
                        to_save=True):
    df_birds = gpd.GeoDataFrame(df_birds, geometry=gpd.points_from_xy(df_birds.x, df_birds.y))

    # Calculate centroids and create interpolation grid
    df_ar['centroid'] = df_ar.geometry.centroid
    x = np.array([pt.x for pt in df_ar.centroid])
    y = np.array([pt.y for pt in df_ar.centroid])
    z = df_ar[variable].values
    xi, yi = np.mgrid[min(x):max(x):resolution*1j, min(y):max(y):resolution*1j]

    # Perform linear interpolation
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    # Plot interpolated data
    fig, ax = plt.subplots(figsize=figsize)#, dpi=dpi)
    cs = ax.contourf(xi, yi, zi, cmap='viridis', levels=1000)

    fig.colorbar(cs, ax=ax)

    if spc:
        # only with the species
        df_birds_spc = df_birds.query('species == @spc')
        df_birds_spc.plot(ax=ax, marker='o', color='red', markersize=5)
        if plot_all_survey_points:
            df_birds.query('species != @spc').plot(ax=ax, marker='o', color='pink', markersize=3)
    else:
        # plot all survey points
        df_birds.plot(ax=ax, marker='o', color='red', markersize=3)

    plt.title(variable + " heatmap", fontsize=20)
    plt.show()


def plot_categorical_feature_on_map(df_ar, df_birds, variable, spc=None, plot_all_survey_points=False, 
                                   buffer_distance=0.01, 
                                   resolution=100,
                                   to_save=True,
                                   cmap='tab10'):

    df_birds = gpd.GeoDataFrame(df_birds, geometry=gpd.points_from_xy(df_birds.x, df_birds.y))

    df_ar[variable] = df_ar[variable].astype('category')

    # Calculate centroids and create interpolation grid
    df_ar['centroid'] = df_ar.geometry.centroid
    x = np.array([pt.x for pt in df_ar.centroid])
    y = np.array([pt.y for pt in df_ar.centroid])
    z = df_ar[variable].astype('category').cat.codes.values
    xi, yi = np.mgrid[min(x):max(x):resolution*1j, min(y):max(y):resolution*1j]

    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    # Plot interpolated data
    fig, ax = plt.subplots(figsize=(10, 10))
    cs = ax.contourf(xi, yi, zi, cmap=cmap, levels=np.arange(len(df_ar[variable].cat.categories)+1)-0.5)
    fig.colorbar(cs, ax=ax, ticks=np.arange(len(df_ar[variable].cat.categories)), format=plt.FuncFormatter(lambda val, loc: df_ar[variable].cat.categories[int(val)]))

    if spc:
        # only with the species
        df_birds_spc = df_birds.query('species == @spc')
        df_birds_spc.plot(ax=ax, marker='o', color='red', markersize=5)
        if plot_all_survey_points:
            df_birds.query('species != @spc').plot(ax=ax, marker='o', color='pink', markersize=3)
    else:
        df_birds.plot(ax=ax, marker='o', color='white', markersize=3)

    # title of variable
    plt.title(variable + " heatmap", fontsize=20)
    if to_save:
        plt.savefig("plots/" + variable + "_heatmap.png")

    else:
        plt.show()

    return fig

def get_spc_info(df_birds, spc):
    df_spc = df_birds.query('species in @spc').reset_index(drop=True)

    dfs = (
        df_spc
        .groupby(['species', 'conservation_rank', 'conservation_status'])
        .size()
        .reset_index(name='number_observations')
        .sort_values(by='number_observations', ascending=False)
        .reset_index(drop=True)
    )

    # get total num of obs in years where species is not None
    # assuming was filtered by years already
    num_total_obs = df_birds.shape[0]

    dfs['percent_in_year'] = 100 * dfs['number_observations'] / num_total_obs
    dfs['percent_in_year'] = dfs['percent_in_year'].round(2)
    return dfs

def process_ndvi(df_gis, df_ar):
    print("Processing NDVI data...")
    df_gis = df_gis.copy()
    df_ar = df_ar.copy()

    df_ndvi = pd.read_csv("data/NDVI_DATA_2017_2020.csv")
    df_ndvi = df_ndvi.drop(columns=["Unnamed: 0"])
    df_ndvi = df_ndvi.rename(columns={"latitude": "y", 
                                      "longitude": "x",
                                      "NDVI": "ndvi"})

    df_ndvi['date'] = pd.to_datetime(df_ndvi['datetime'])
    # without hours and minutes
    df_ndvi['date'] = df_ndvi['date'].dt.date
    df_ndvi['date'] = pd.to_datetime(df_ndvi['date'])
    # remove datetime col
    df_ndvi = df_ndvi.drop(columns=["datetime"])

    df_ndvi_year = (
        df_ndvi
        .assign(year_prev = lambda x: x.date.dt.year)
        .assign(year = lambda x: x.date.dt.year + 1) # year to join to
        .groupby(["y", "x", "year"])
        .mean("ndvi")
        .reset_index(drop=False)
    )

    def find_closest_ndvi(row, df_ndvi, by_year=False):
        x_diff = np.abs(df_ndvi['x'] - row['x'])
        y_diff = np.abs(df_ndvi['y'] - row['y'])
        if by_year:
            date_diff = np.abs(df_ndvi['year'] - row['year'])
        else:
            date_diff = np.abs((df_ndvi['date'] - row['date']).dt.days)
        total_diff = y_diff + x_diff + date_diff
        return df_ndvi.loc[total_diff.idxmin(), 'ndvi']


    df_gis['date'] = pd.to_datetime(df_gis['date'])

    if DEBUG:
        df_gis = df_gis.sample(frac=0.1, random_state=42).reset_index(drop=True)
        df_ar = df_ar.sample(frac=0.1, random_state=42).reset_index(drop=True)

    print("Processing NDVI data for survey...")

    df_gis_ndvi = df_gis.copy()
    df_gis_ndvi['ndvi'] = df_gis_ndvi.apply(find_closest_ndvi, args=(df_ndvi, False), axis=1)
    df_gis_ndvi['ndvi_prev_year'] = df_gis_ndvi.apply(find_closest_ndvi, args=(df_ndvi_year, True), axis=1)
    df_gis_ndvi.to_csv("./data/birds_survey_gis_ndvi.csv", index=False)

    print("Processing NDVI data for AR...")

    df_ar['date'] = df_gis_ndvi['date'].max()
    df_ar['date'] = pd.to_datetime(df_ar['date'])
    df_ar['year'] = df_ar['date'].dt.year

    df_ar['ndvi'] = df_ar.apply(find_closest_ndvi, args=(df_ndvi, False), axis=1)
    df_ar['ndvi_prev_year'] = df_ar.apply(find_closest_ndvi, args=(df_ndvi_year, True), axis=1)

    for col in df_ar.columns:
        if df_ar[col].dtype == 'object':
            df_ar[col] = df_ar[col].astype(str)

    df_ar.to_file("data/df_ar_sub_ndvi.geojson")
    df_ndvi.to_csv("data/df_ndvi.csv", index=False)

    return df_gis_ndvi, df_ar, df_ndvi



def cross_validate_per_species(model_class, df, species_list, cfg, test_size=0.2):
    df = df.copy()
    results = pd.DataFrame(
        columns=['species', 
                 'num_samples_total',
                 'num_samples_train',
                 'num_samples_val',
                 'train_auc', 'val_auc',
                 'train_precision', 'val_precision',
                 'train_recall', 'val_recall',
                    'train_log_loss', 'val_log_loss'  
                 ]
    )

    # Iterate over each species
    for species in species_list:
        print(species)
        # modify cfg
        cfg['species'] = [species]

        # this preprocessing doesn't involve functions which 
        # need to be computed separately for train, it is later in the fit
        X, y, _ = preproc_for_model(df, None, cfg=cfg)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=SEED)

        model = model_class(cfg=cfg)
        model.fit(X_train, y_train)

        train_auc = roc_auc_score(y_train, model.predict_proba(X_train))
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val))

        train_log_loss = log_loss(y_train, model.predict_proba(X_train).astype(np.float64))
        val_log_loss = log_loss(y_val, model.predict_proba(X_val).astype(np.float64))

        # Compute precision and recall for train set
        y_train_pred = model.predict(X_train)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)

        # Compute precision and recall for validation set
        y_val_pred = model.predict(X_val)
        val_precision = precision_score(y_val, y_val_pred)
        val_recall = recall_score(y_val, y_val_pred)

        # num samples refers to the positive class, i.e. the target species
        num_train = np.sum(y_train == 1)
        num_val = np.sum(y_val == 1)
        num_total = num_train + num_val

        # Record results
        results = results.append({
            'species': species,
            'num_samples_total': num_total,
            'num_samples_train': num_train,
            'num_samples_val': num_val,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'train_precision': train_precision,
            'val_precision': val_precision,
            'train_recall': train_recall,
            'val_recall': val_recall,
            'train_log_loss': train_log_loss,
            'val_log_loss': val_log_loss
        }, ignore_index=True)

    return results


def train_and_plot_roc_curve(models, df, species, test_size=0.4, random_state=None):

    plt.figure()

    for model_dict_iter in models.items():
        model_dict = model_dict_iter[1]
        cfg = model_dict['cfg']
        cfg['species'] = [species]
        X, y, _ = preproc_for_model(df.copy(), None, cfg=cfg)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model_class = model_dict['model_class']
        model = model_class(cfg=cfg)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        model_name = model_class.__name__.replace('ModelBird', '')
        plt.plot(fpr, tpr, lw=2, label='{} ({:.2f})'.format(model_name, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for {}'.format(cfg['species'][0][::-1]))
    plt.legend(loc="lower right")
    plt.show()

    return model


def train_and_plot_precision_recall_curve(models, df, species, test_size=0.4, random_state=None):

    plt.figure()

    for model_dict_iter in models.items():
        model_dict = model_dict_iter[1]
        cfg = model_dict['cfg']
        cfg['species'] = [species]
        X, y, _ = preproc_for_model(df.copy(), None, cfg=cfg)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model_class = model_dict['model_class']
        model = model_class(cfg=cfg)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_val)

        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)

        model_name = model_class.__name__.replace('ModelBird', '')

        auc = average_precision_score(y_val, y_pred_proba)

        plt.plot(recall, precision, lw=2, label='{} ({:.2f})'.format(model_name, auc))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve for {}'.format(cfg['species'][0][::-1]))
    plt.legend(loc="lower right")
    plt.show()

    return model






