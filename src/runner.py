
from src.utils import *

df_spc, df_cls, df_out, feature_names, reserves = load_data()
feature_types = infer_feature_types(df_spc[feature_names])

features = {}

features['cont'] = [f for f in feature_names if feature_types[f] == 'Continuous']
features['cat'] = [f for f in feature_names if feature_types[f] == 'Categorical']

spc = "אלימון"

cfg = {
    'species': [spc],
    'features_cont': features['cont'],
    'features_cat': features['cat'],
    'survey_years': [2020, 2019, 2018]
}
cfg['features'] = features['cont'] + features['cat']

drop_cats = {'reserve_states': 'not_reserved'}

if cfg['features_cat']:
    to_ohe = True
else:
    to_ohe = False

model = ModelBirdLogisticRegression(to_scale=True,
                                    to_ohe=to_ohe,
                                    cfg=cfg)

res = run_exp(model,
              df_cls,
              df_out,
              cfg=cfg,
              with_arava_preds=False)

print("ran model")