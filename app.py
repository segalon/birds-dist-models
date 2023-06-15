
from src.models import *
from src.utils import *
from src.config import *

import streamlit as st
import os 

path = "/Users/user/projects-uni/birds-conservation"
os.chdir(path)


def plot_feature_relevance(model, model_name):
    """
    Plot the feature relevance of the model.
    """
    if  model_name in ["CatBoost", "MaxEnt"]:
        return model.plot_feature_importances()
    elif model_name == "Logistic Regression":
        return plot_dot_whisker(model.model)

def process_and_display_results(cfg, df_cls, df_ar, df_birds, probas_list, aggregation_type, models_list=None):

    years = cfg['survey_years']

    df_birds = df_birds.query('year in @years')
    df_spc_info = get_spc_info(df_birds, cfg['species'])

    st.table(df_spc_info)

    fig_map, ax_map = plot_probas_on_map(
                    df_ar=df_ar,
                    df_birds=df_birds.query('year in @years'),
                    probas_list=probas_list,
                    spc_list=cfg['species'],
                    resolution=500,
                    aggregation_type=aggregation_type,
                    plot_other_species=True,
                    plot_nature_reserves=plot_nature_reserves,
                    shm_negev=shm_negev)
   
    st.pyplot(fig_map)
    if plot_feature_importance:
        if len(models_list) == 1:
            # for now only plot if one model, because shap values
            # can be expensive to compute
            for model in models_list: 
                fig_fr = plot_feature_relevance(model, cfg['model_name'])
                st.pyplot(fig_fr)


st.title("Birds distributions")

df_birds, df_cls, df_ar, shm_negev = load_data()

min_obs = 3
df_birds = df_birds.groupby('species').filter(lambda x: len(x) >= min_obs)

features = {
    'cont': {
        'climatic': ['mean_temp_jan', 'max_temp_june', 'mean_temp_aug', 'min_temp_jan', 'mean_annual_rainfall',
                     'seasonal_temp_range'],
        'ndvi': ['ndvi', 'ndvi_prev_year'],
        'distances': ['dist_to_reserve', 'dist_to_settlement', 'dist_to_road', 'dist_to_sea'],
        'topography': ['avg_elevation_100m2', 'slope', 'rezef'],
        #'topography': ['avg_elevation_100m2', 'slope'],
        'human_impact': ['human_impact'],
        'vegetation': ['veg_cover'],
    },
    'cat': {
        'climatic': [],
        'soil': ['soil_name', 'rock_name'],
        'distances': [],
        'topography': ['aspect_name', 'corridor'],
        'human_impact': [],
        'ecosystem': ['ecosystem_name'],
        #'survey': ['year', 'is_nature_reserve', 'reserve_status', 'is_fire_zone'],
        #'survey': ['year'], # only if more then one year is chosen this is used

    }
}


variables = features.copy()

# -------- Streamlit code --------

# select model
available_models = ['CatBoost', 'Logistic Regression', 'MaxEnt']
selected_model = st.selectbox("Select model", available_models)


# select model class
model_class = None
if selected_model == 'CatBoost':
    model_class = ModelBirdCatBoost
elif selected_model == 'Logistic Regression':
    model_class = ModelBirdLogisticRegression
elif selected_model == 'MaxEnt':
    model_class = ModelBirdMaxEnt



# filter the available species by the selected years
available_years = df_birds['year'].unique()
selected_years = st.multiselect(
    'Select survery years',
    available_years)


if len(selected_years) == 0:
    #st.write("Please select years")
    st.stop()


df_birds = df_birds.query('year in @selected_years')


available_ranks = df_birds['conservation_status'].unique()
container_ranks = st.container()
all_ranks = st.checkbox("Select all conservation ranks")

if all_ranks:
    selected_ranks = container_ranks.multiselect("Select conservation rank:", 
        available_ranks, available_ranks)
else:
    selected_ranks =  container_ranks.multiselect("Select conservation rank:",
        available_ranks)
    
if len(selected_ranks) == 0:
    #st.write("Please select conservation ranks")
    st.stop()


# filter the available species by the selected years
available_species = (
    df_birds
    .query('year in @selected_years')
    .query('conservation_status in @selected_ranks')
)

available_species = list(available_species['species'].unique())
available_species.sort()


container_species = st.container()
all_species = st.checkbox("Select all species")
if all_species:
    selected_species = container_species.multiselect("Select species", 
        available_species, available_species)  
else:
    selected_species =  container_species.multiselect("Select species",
        available_species)

# if not selected then say "please select species"
if len(selected_species) == 0:
    #st.write("Please select species")
    st.stop()

st.subheader("Select continuous variables")

climatic_container = st.container()
all_climatic_variables = st.checkbox("Select all climatic variables")
if all_climatic_variables:
    selected_climatic_variables = climatic_container.multiselect("Climatic variables (not at the time of the survey)", 
                                             variables['cont']['climatic'], 
                                             variables['cont']['climatic'])
else:
    selected_climatic_variables = climatic_container.multiselect("Climatic variables (not at the time of the survey)", 
                                             variables['cont']['climatic'])

# NDVI variables
ndvi_container = st.container()
all_ndvi_variables = st.checkbox("Select all NDVI variables", value=True)
if all_ndvi_variables:
    selected_ndvi_variables = ndvi_container.multiselect("NDVI variables", 
                                         variables['cont']['ndvi'], 
                                         variables['cont']['ndvi'])
else:
    selected_ndvi_variables = ndvi_container.multiselect("NDVI variables", 
                                         variables['cont']['ndvi'])

# Distance variables
distances_container = st.container()
# by default all distance variables are selected
all_distances_variables = st.checkbox("Select all distance variables", value=True)
if all_distances_variables:
    selected_distances_variables = distances_container.multiselect("Distance variables", 
                                              variables['cont']['distances'], 
                                              variables['cont']['distances'])
else:
    selected_distances_variables = distances_container.multiselect("Distance variables", 
                                              variables['cont']['distances'])

# Topography variables
topography_container = st.container()
all_topography_variables = st.checkbox("Select all topography variables", value=True)
if all_topography_variables:
    selected_topography_variables = topography_container.multiselect("Topography variables", 
                                               variables['cont']['topography'], 
                                               variables['cont']['topography'])
else:
    selected_topography_variables = topography_container.multiselect("Topography variables", 
                                               variables['cont']['topography'])

veg_container = st.container()
all_veg_variables = st.checkbox("Select all vegetation variables", value=True)
if all_veg_variables:
    selected_veg_variables = veg_container.multiselect("Vegetation variables", 
                                               variables['cont']['vegetation'],
                                               variables['cont']['vegetation'])

# Human impact variables
human_impact_container = st.container()
all_human_impact_variables = st.checkbox("Select all human impact variables")
if all_human_impact_variables:
    selected_human_impact_variables = human_impact_container.multiselect("Human impact variables", variables['cont']['human_impact'], variables['cont']['human_impact'])
else:
    selected_human_impact_variables = human_impact_container.multiselect("Human impact variables", variables['cont']['human_impact'])


variables_cont = selected_climatic_variables + selected_ndvi_variables + selected_distances_variables + selected_topography_variables + selected_human_impact_variables

st.write("Selected categorical variables:")


models_with_cats = ['CatBoost', 'MaxEnt']
if selected_model in models_with_cats:
    value = True
else:
    value = False


variables = features
variables_cat = variables['cat']

# soil
soil_container = st.container()

all_soil_variables = st.checkbox("Select all soil variables", 
                                 value=value)
if all_soil_variables:
    selected_soil_variables = soil_container.multiselect("Soil variables", variables_cat['soil'], variables_cat['soil'])
else:
    selected_soil_variables = soil_container.multiselect("Soil variables", variables_cat['soil'])

# land use

# topography_cat
topography_cat_container = st.container()
all_topography_cat_variables = st.checkbox("Select all cat topography variables", value=value)
if all_topography_cat_variables:
    selected_topography_cat_variables = topography_cat_container.multiselect("Topography variables", variables_cat['topography'], variables_cat['topography'])
else:
    selected_topography_cat_variables = topography_cat_container.multiselect("Topography variables", variables_cat['topography'])

# ecosystem
ecosystem_container = st.container()
all_ecosystem_variables = st.checkbox("Select all ecosystem variables", value=value)
if all_ecosystem_variables:
    selected_ecosystem_variables = ecosystem_container.multiselect("Ecosystem variables", variables_cat['ecosystem'], variables_cat['ecosystem'])
else:
    selected_ecosystem_variables = ecosystem_container.multiselect("Ecosystem variables", variables_cat['ecosystem'])

variables_cat = selected_soil_variables + selected_topography_cat_variables + selected_ecosystem_variables



# for plotting nature reserves 
plot_nature_reserves = st.checkbox("Plot nature reserves", value=False)
plot_feature_importance = st.checkbox("Plot feature importance", value=False)


# -------- / Streamlit code --------


cfg = {
    'species': selected_species,
    'features': variables_cont + variables_cat,
    'features_cont': variables_cont,
    'features_cat': variables_cat,
    'survey_years': selected_years,
    'model_name': selected_model,
}


years = cfg['survey_years']


aggregation_type = "Treat separately"


if aggregation_type == "Group together":
    #model = ModelBirdLogisticRegression(to_scale=True, to_ohe=False, cfg=cfg)

    model= model_class(to_scale=True, to_ohe=False, cfg=cfg)

    res = run_exp(model, df_cls, df_ar, cfg=cfg)
    probas_list = [res['y_pred_arv']]
    models_list = None
else:  # Treat separately
    probas_list = []
    models_list = []
    for species in selected_species:
        cfg_single_species = cfg.copy()
        cfg_single_species['species'] = [species]
        # if model_class == ModelBirdLogisticRegression:
        model_single_species = model_class(to_scale=True, cfg=cfg_single_species)
        res_single_species = run_exp(model_single_species, df_cls, df_ar, cfg=cfg_single_species)
        probas_list.append(res_single_species['y_pred_arv'])
        models_list.append(model_single_species)
    probas_list = np.array(probas_list)


process_and_display_results(cfg, df_cls, df_ar, df_birds, probas_list, aggregation_type, models_list)

