import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import lightgbm as lgb
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle
from matplotlib import cm
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from matplotlib.patches import Arc
import shap

# Définition imputer et scaler
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
scaler = preprocessing.MinMaxScaler()

# Chemins
data_path = './preprocessed_data/'
model_path = './saved_model/'

# Charger le fichier de données
df_test = pd.read_csv(data_path+'df_test_compressed.gzip', compression='gzip')
df_test.drop('Unnamed: 0', axis=1, inplace=True)

def predictions(data):
    # Copie du fichier de base
    df_dashboard = data.copy()

    # Définition ID clients
    id_client = df_dashboard['SK_ID_CURR']

    # Définition liste features
    features_names = df_dashboard.columns.to_list()

    # Imputation, encodage et prédictions
    X_test = df_dashboard
    X_test = df_dashboard.drop('SK_ID_CURR', axis=1)
    X_test_filled = imputer.fit_transform(X_test)
    X_test_filled_scaled = scaler.fit_transform(X_test_filled)
    y_proba = model.predict_proba(X_test_filled_scaled)[:, 1]
    y_pred = model.predict(X_test_filled_scaled)



# Charger le modèle à partir du fichier pickle
with open(model_path+'LightGBM_smote_tuned.pckl', 'rb') as f:
    model = pickle.load(f)

# Appel de la fonction de prédictions
predictions(df_test)

# Constitution du dataframe pour merge
df_score = pd.DataFrame({'SK_ID_CURR' : id_client,
                         'PRED_CLASSE_CLIENT' : y_pred,
                         'SCORE_CLIENT' : y_proba,
                         'SCORE_CLIENT_%' : np.round(y_proba * 100, 1)})

# Ajout des prédictions au dataframe du jeu de test
df_dashboard = df_score.merge(df_test, on='SK_ID_CURR', how='left')

df_infos = df_test[['SK_ID_CURR',
                    'AMT_INCOME_TOTAL',
                    'CNT_CHILDREN']]

# Ajout des variables manquantes au dataframe du dashboard
df_dashboard = df_dashboard.merge(df_infos, on='SK_ID_CURR', how='right')

df_dash = imputer.fit_transform(df_dashboard)

df_dashboard_final = pd.DataFrame(df_dash, columns=df_dashboard.columns)
df_dashboard_final['SK_ID_CURR'] = df_dashboard_final['SK_ID_CURR'].astype(int)

# Explainer SHAP (top25 features)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_filled_scaled)
expected_value = explainer.expected_value
vals =  np.abs(shap_values[1]).mean(0)
df_feature_importance = pd.DataFrame(list(zip(features_names, vals)),
                                     columns=['Features','Features_importance_shapley'])
df_feature_importance.sort_values(by=['Features_importance_shapley'],
                                  ascending=False,
                                  inplace=True)
df_feature_importance_25 = df_feature_importance.head(25)

# Définition des features de score et prédiction
other_features = ['SK_ID_CURR',
                  'PRED_CLASSE_CLIENT',
                  'SCORE_CLIENT',
                  'SCORE_CLIENT_%']

# Définition des features du dataframe final
selected_columns = other_features + df_feature_importance_25['Features'].tolist()
df_dashboard_final = df_dashboard_final[selected_columns]

# Renommer les features restantes (ENG => FR)
df_dashboard_final = df_dashboard_final.rename(columns={
     'SK_ID_CURR' : 'ID client',
     'PRED_CLASSE_CLIENT' : 'Prédiction crédit',
     'SCORE_CLIENT' : 'Score client',
     'SCORE_CLIENT_%' : 'Score client (sur 100)',
     'BUREAU_MONTHS_BALANCE_SIZE_SUM': 'Montant mensuel agence (somme)',
     'INSTAL_DAYS_PAST_DUE_MAX': 'Somme maximale dûe',
     'REGION_RATING_CLIENT' : 'Notation client région',
     'EXT_SOURCE_1' : 'Score interne 1',
     'EXT_SOURCE_2' : 'Score interne 2',
     'LIVE_CITY_NOT_WORK_CITY' : 'Ville travail différente ville résidence',
     'NAME_CONTRACT_TYPE' : 'Type contrat',
     'ANNUITY_INCOME_PERCENT' : 'Part annuités sur revenu total (%)',
     'PREV_DAYS_DECISION_MEAN' : 'Nb jours prise de décision',
     'DAYS_BIRTH' : 'Âge',
     'NAME_EDUCATION_TYPE_Academic degree' : 'Diplôme etude supérieure',
     'CODE_GENDER' : 'Genre',
     'AMT_CREDIT' : 'Montant crédit',
     'NAME_FAMILY_STATUS_Civil marriage' : 'Statut marié',
     'BUREAU_CREDIT_DAY_OVERDUE_MEAN' : 'Retard de paiement (jours)',
     'BUREAU_DAYS_CREDIT_ENDDATE_MIN' : 'Temps restant crédit (jours)',
     'DAYS_REGISTRATION' : 'Ancienneté client (jours)',
     'BUREAU_AMT_CREDIT_SUM_DEBT_MAX' : 'Débit maximal bureau local'
})

# Modification des variables catégorielles et booléennes
df_dashboard_final['Genre'] = df_dashboard_final['Genre'].replace({0: 'H', 1: 'F'})
df_dashboard_final['Statut marié'] = df_dashboard_final['Statut marié'].replace({0: 'Non', 1: 'Oui'})
df_dashboard_final['Prédiction crédit'] = \
    df_dashboard_final['Prédiction crédit'].replace({0: 'Non défaillant', 1: 'Défaillant'})
df_dashboard_final['Ville travail différente ville résidence'] = \
    df_dashboard_final['Ville travail différente ville résidence'].replace({0: 'Non', 1: 'Oui'})
df_dashboard_final['Type contrat'] = \
    df_dashboard_final['Type contrat'].replace({0: 'Cash loan', 1: 'Revolving loan'})
df_dashboard_final['Diplôme etude supérieure'] = \
    df_dashboard_final['Diplôme etude supérieure'].replace({0: 'Non', 1: 'Oui'})

# Valeurs absolues et arrondis
df_dashboard_final['Temps restant crédit (jours)'] = \
    df_dashboard_final['Temps restant crédit (jours)'].abs()
df_dashboard_final['Ancienneté client (jours)'] = \
  df_dashboard_final['Ancienneté client (jours)'].abs()
df_dashboard_final['Nb jours prise de décision'] = \
    df_dashboard_final['Nb jours prise de décision'].abs()
df_dashboard_final['Âge'] = round(df_dashboard_final['Âge'].abs()/365, 0)
df_dashboard_final['Score client'] = round(df_dashboard_final['Score client'], 4)
df_dashboard_final['Score interne 1'] = round(df_dashboard_final['Score interne 1'], 4)
df_dashboard_final['Score interne 2'] = round(df_dashboard_final['Score interne 2'], 4)
df_dashboard_final['Part annuités sur revenu total (%)'] = \
    round(df_dashboard_final['Part annuités sur revenu total (%)'], 4)

# Enregistrement du fichier final pour le dashboard Streamlit
df_dashboard_final.to_csv('df_dashboard_final.csv')

rename_mapping = {
     'BUREAU_MONTHS_BALANCE_SIZE_SUM': 'Montant mensuel agence (somme)',
     'INSTAL_DAYS_PAST_DUE_MAX': 'Somme maximale dûe',
     'REGION_RATING_CLIENT' : 'Notation client région',
     'EXT_SOURCE_1' : 'Score interne 1',
     'EXT_SOURCE_2' : 'Score interne 2',
     'LIVE_CITY_NOT_WORK_CITY' : 'Ville travail différente ville résidence',
     'NAME_CONTRACT_TYPE' : 'Type contrat',
     'ANNUITY_INCOME_PERCENT' : 'Part annuités sur revenu total (%)',
     'PREV_DAYS_DECISION_MEAN' : 'Nb jours prise de décision',
     'DAYS_BIRTH' : 'Âge',
     'NAME_EDUCATION_TYPE_Academic degree' : 'Diplôme etude supérieure',
     'CODE_GENDER' : 'Genre',
     'AMT_CREDIT' : 'Montant crédit',
     'NAME_FAMILY_STATUS_Civil marriage' : 'Statut marié',
     'BUREAU_CREDIT_DAY_OVERDUE_MEAN' : 'Retard de paiement (jours)',
     'BUREAU_DAYS_CREDIT_ENDDATE_MIN' : 'Temps restant crédit (jours)',
     'DAYS_REGISTRATION' : 'Ancienneté client (jours)',
     'BUREAU_AMT_CREDIT_SUM_DEBT_MAX' : 'Débit maximal bureau local'
}
df_feature_importance_25['Features'] = df_feature_importance_25['Features'].replace(rename_mapping)
df_feature_importance_25.to_csv('df_feature_importance_25.csv')