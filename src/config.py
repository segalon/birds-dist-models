
import copy

data_wt_gis_path = "data/birds_survey_gis_ndvi.csv"

species_to_keep_5 = ["אלימון", "חנקן גדול", "מדברון", "צחיחנית מדבר", "קטה סנגלית"]

vars_gis = ['point_ITM', 'cellId', 'cell_coords',
    'rainboth', 'mdt1', 'mdt8', 'mmmax6IN', 'mmmin1ST', 'topo100', 'slope',
    'D2RSV', 'D2STTL', 'D2RD', 'D2SEA', 'anudiff', 'rockID', 'rock_name',
    'rock_name_heb', 'corridor', 'ecosystemID', 'ecosystem_name',
    'ecosystem_name_heb', 'human_impact', 'landuseID', 'landuse_name',
    'landuse_name_heb', 'aspectID', 'aspect_name', 'aspect_name_heb',
    'soilID', 'soil_name', 'soil_name_heb', 'landcoverID',
    'land_cover_name', 'land_cover_name_heb', 'rezef', 'veg_cover']

vars_ndvi = ['ndvi', 'ndvi_prev_year']

# description of the variables in their original names
vars_gis_desription = {
    'point_ITM': 'The point in the coordinate system',
    'cell_coords': 'The cell coordinates',
    'rainboth': 'The average annual rainfall',
    'mdt1': 'The mean temperature in january',
    'mdt8': 'The mean temperature in august',
    'mmax6IN': 'The maximum temperature in june',
    'mmin1ST': 'The minimum temperature in january',
    'topo100': 'elevation averages of 100m x 100m cells',
    'slope': 'slope',
    'rock_name': 'rock name',
    'anudiff': 'seasonal temperature range',
    'corridor': 'ecological corridor',
    'ecosystem_name': 'ecosystem_name',
    'human_impact': 'human impact',
    'aspect_name': 'aspect_name', 
    'soil_name': 'soil_name',
    'land_cover_name': 'land_cover_name',
    'rezef': 'continuity of open areas',
    'veg_cover': 'vegetation cover' ,
    'D2RSV': 'distance to the nearest reserve',
    'D2STTL': 'distance to the nearest settlement',
    'D2RD': 'distance to the nearest road',
    'D2SEA': 'distance to the nearest sea'
    }


vars_eco = ['rainboth', 'mdt1', 'mdt8', 'mmmax6IN', 'mmmin1ST', 'topo100', 'slope',
    'anudiff', 'rockID', 'corridor', 'ecosystemID', 'landuseID', 'aspectID', 'soilID',
    'landcoverID', 'rezef', 'veg_cover']


vars_survey = [
    'date', 'year', 'y', 'x', 'obs_type', 'presence', 
    'from_hour', 'to_hour', 'species', 'amount', 'distance', 
    'conservation_status', 'is_fire_zone', 'is_nature_reserve', 
    'reserve_name', 'reserve_status']


vars_gis_new_names = {
    'point_ITM': 'point_ITM',
    'cell_coords': 'cell_coords',
    'rainboth': 'mean_annual_rainfall',
    'mdt1': 'mean_temp_jan',
    'mdt8': 'mean_temp_aug',
    'mmmax6IN': 'max_temp_june',
    'mmmin1ST': 'min_temp_jan',
    'topo100': 'avg_elevation_100m2',
    'slope': 'slope',
    'rock_name': 'rock_name',
    'anudiff': 'seasonal_temp_range',
    'corridor': 'corridor',
    'ecosystem_name': 'ecosystem_name',
    'human_impact': 'human_impact',
    'aspect_name': 'aspect_name',
    'soil_name': 'soil_name',
    'land_cover_name': 'land_cover_name',
    'rezef': 'rezef',
    'veg_cover': 'veg_cover',
    'D2RSV': 'dist_to_reserve',
    'D2STTL': 'dist_to_settlement',
    'D2RD': 'dist_to_road',
    'D2SEA': 'dist_to_sea',
}

vars_gis_selection = list(vars_gis_new_names.values())

vars_gis_cont = ['mean_annual_rainfall', 
                 'mean_temp_jan', 
                 'max_temp_june', 
                 'mean_temp_aug', 
                 'min_temp_jan',
                 'avg_elevation_100m2',
                 'slope', 
                 'dist_to_reserve', 
                 'dist_to_settlement',
                 'dist_to_road', 
                 'dist_to_sea', 
                 'seasonal_temp_range', 
                 'rezef',
                 'human_impact', 
                 'veg_cover']


vars_gis_cat = ['corridor', 'rock_name', 'ecosystem_name', 'aspect_name', 'soil_name']

vars_ndvi = ['ndvi', 'ndvi_prev_year']

features = {
    'cont': {
        'climatic': ['mean_temp_jan', 'max_temp_june', 'mean_temp_aug', 'min_temp_jan', 'mean_annual_rainfall',
                     'seasonal_temp_range'],
        'ndvi': ['ndvi', 'ndvi_prev_year'],
        'distances': ['dist_to_reserve', 'dist_to_settlement', 'dist_to_road', 'dist_to_sea'],
        'topography': ['avg_elevation_100m2', 'slope', 'rezef'],
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
    }
}


features_cat = features['cat'].values()
features_cat = [item for sublist in features_cat for item in sublist]

features_cont = features['cont'].values()
features_cont = [item for sublist in features_cont for item in sublist]

cfg_catboost = {
    'features_cont': features_cont,
    'features_cat': features_cat,
    'survey_years': [2018, 2019, 2020],
}

