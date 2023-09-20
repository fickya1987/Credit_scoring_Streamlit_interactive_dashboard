import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metrics import DataDriftTable, DatasetDriftMetric


# Ouverture du dataset
path = './preprocessed_data/'

# Définition des features
data_ref = pd.read_csv(path+'df_train.csv')
data_ref = data_ref[['REGION_RATING_CLIENT',
                     'EXT_SOURCE_1',
                     'EXT_SOURCE_2',
                     'LIVE_CITY_NOT_WORK_CITY',
                     'NAME_CONTRACT_TYPE',
                     'ANNUITY_INCOME_PERCENT',
                     'AMT_REQ_CREDIT_BUREAU_QRT',
                     'PREV_DAYS_DECISION_MEAN',
                     'DAYS_BIRTH',
                     'BUREAU_MONTHS_BALANCE_SIZE_SUM',
                     'BUREAU_DAYS_CREDIT_ENDDATE_MEAN',
                     'INSTAL_DAYS_PAST_DUE_SUM',
                     'INSTAL_DAYS_PAST_DUE_MAX',
                     'NAME_EDUCATION_TYPE_Academic degree',
                     'INSTAL_PAYMENT_PERCENT_VAR',
                     'CODE_GENDER',
                     'AMT_CREDIT']]

data_cur = pd.read_csv(path+'df_test.csv')
data_cur = data_cur[['REGION_RATING_CLIENT',
                     'EXT_SOURCE_1',
                     'EXT_SOURCE_2',
                     'LIVE_CITY_NOT_WORK_CITY',
                     'NAME_CONTRACT_TYPE',
                     'ANNUITY_INCOME_PERCENT',
                     'AMT_REQ_CREDIT_BUREAU_QRT',
                     'PREV_DAYS_DECISION_MEAN',
                     'DAYS_BIRTH',
                     'BUREAU_MONTHS_BALANCE_SIZE_SUM',
                     'BUREAU_DAYS_CREDIT_ENDDATE_MEAN',
                     'INSTAL_DAYS_PAST_DUE_SUM',
                     'INSTAL_DAYS_PAST_DUE_MAX',
                     'NAME_EDUCATION_TYPE_Academic degree',
                     'INSTAL_PAYMENT_PERCENT_VAR',
                     'CODE_GENDER',
                     'AMT_CREDIT']]


# Génération du dashboard de DataDrift Evidently
data_drift_dataset_report = Report(metrics=[
    DatasetDriftMetric(),
    DataDriftTable()
])

data_drift_dataset_report.run(reference_data=data_ref, current_data=data_cur)
data_drift_dataset_report

# Enregistrement du dashboard de DataDrift Evidently
data_drift_dataset_report.save_html("Datadrift_report.html")