

# export PYTHONPATH="/Users/user/projects-uni/birds-conservation"

from src.models import *
from src.utils import *
from src.config import *

from src.config import features

df_birds, df_cls, df_ar = load_data()


features_cont = features['cont']['ndvi'] + \
    features['cont']['topography']

features_cat = features['cat']['soil'] + \
    features['cat']['topography']

cfg = {
    'species': [species_to_keep[0]],
    'features_cont': features_cont,
    'features_cat': features_cat,
    'survey_years': [2020, 2019, 2018]

}

cfg['features'] = cfg['features_cat'] + cfg['features_cont']


X_train, y_train, df_ar_t = preproc_for_model(df_cls, df_ar, cfg=cfg)

model = CatBoostClassifier(verbose=False)
model.fit(X_train, y_train, cat_features=features_cat)
y_pred_arava = model.predict_proba(df_ar_t)[:, 1]

years = cfg['survey_years']

plot_probas_on_map(df_ar,
                    df_birds=df_birds.query('year in @years'),
                    probas_list=[y_pred_arava],
                    spc_list=cfg['species'],
                    plot_all_survey_points=False,
                    resolution=500)
# plot on arava

# load from csv
# read as geopandas
# df_ar = gpd.read_file('./data/df_ar_sub.geojson')
# # convert to gdf
# df_ar = gpd.GeoDataFrame(df_ar, geometry='geometry')
# X_arv = df_ar[features]

# predict on arava
#y_pred_arv = model.predict_proba(X_arv)

# on map
#df_birds = df_survey[df_survey.species == spc]



# res = dict()
# res['y_pred_train'] = y_pred_train
# res['y_train'] = y_train

# p = plot_hist_pred_probas(res, spc)

# p = plot_probas_on_map(df_ar, df_birds, y_pred_arv)
# # save ithe ggplot as image
# p.save('./plots/plot_probas_tr.png')

