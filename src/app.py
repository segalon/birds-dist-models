
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
    if model_name in ["CatBoost", "MaxEnt"]:
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


st.title("Species distribution model")

df_birds, df_cls, df_ar, shm_negev = load_data()

min_obs = 3
df_birds = df_birds.groupby('species').filter(lambda x: len(x) >= min_obs)

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


cont_vars_container = st.container()
all_cont_vars = st.checkbox("Select all continuous variables")
if all_cont_vars:
    selected_cont_vars = cont_vars_container.multiselect("Continuous variables", variables['cont'], variables['cont'])
else:
    selected_cont_vars = cont_vars_container.multiselect("Continuous variables", variables['cont'])

variables_cont = selected_cont_vars

st.write("Selected categorical variables:")


models_with_cats = ['CatBoost', 'MaxEnt']
if selected_model in models_with_cats:
    value = True
else:
    value = False


variables = features
variables_cat = variables['cat']

#variables_cat = selected_soil_variables + selected_topography_cat_variables + selected_ecosystem_variables



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
    model = model_class(to_scale=True, to_ohe=False, cfg=cfg)
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
