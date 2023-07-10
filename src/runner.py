
from src.models import *
from src.utils import *
from src.config import *


df_spc, df_cls, df_out, feature_names, reserves = load_data()

features = {
    'cont': feature_names,
    'cat': []
}

spc = "אלימון"

cfg = {
    'species': [spc],
    'features_cont': features['cont'],
    'features_cat': features['cat'],
    'survey_years': [2020, 2019, 2018]
}

drop_cats = {'reserve_states': 'not_reserved'}
# model = ModelBirdLogisticRegression(to_scale=True,
#                                     to_ohe=True,
#                                     cfg=cfg,
#                                     drop_categories=drop_cats,
#                                     default_drop="first")

model = ModelBirdLogisticRegression(to_scale=True,
                                    to_ohe=True,
                                    cfg=cfg)
res = run_exp(model,
              df_cls,
              df_out,
              spc=[spc],
              cfg=cfg,
              with_arava_preds=False)

print("ran model")