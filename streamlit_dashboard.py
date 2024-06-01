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

# Fungsi terkait dengan pengukur risiko pelanggan
def degree_range(n):
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points


def rot_text(ang):
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation


def gauge(arrow=0.5, labels=['Rendah', 'Sedang', 'Tinggi', 'Sangat tinggi'],
          title='', min_val=0, max_val=100, threshold=-1.0,
          colors='RdYlGn_r', n_colors=-1, ax=None, figsize=(3, 2)):
    N = len(labels)
    n_colors = n_colors if n_colors > 0 else N
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, n_colors)
        cmap = cmap(np.arange(n_colors))
        colors = cmap[::-1]
    if isinstance(colors, list):
        n_colors = len(colors)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ang_range, _ = degree_range(n_colors)

    for ang, c in zip(ang_range, colors):
        ax.add_patch(Wedge((0., 0.), .4, *ang, width=0.10, facecolor='w', lw=2, alpha=0.5))
        ax.add_patch(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=0, alpha=0.8))

    _, mid_points = degree_range(N)
    labels = labels[::-1]
    a = 0.45
    for mid, lab in zip(mid_points, labels):
        ax.text(a * np.cos(np.radians(mid)), a * np.sin(np.radians(mid)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=12, \
                fontweight='bold', rotation=rot_text(mid))

    ax.add_patch(Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2))
    ax.text(0, -0.10, title, horizontalalignment='center', verticalalignment='center', fontsize=20, fontweight='bold')

    # Ambang batas pengukur
    if threshold > min_val and threshold < max_val:
        pos = 180 * (max_val - threshold) / (max_val - min_val)
        a = 0.25;
        b = 0.18;
        x = np.cos(np.radians(pos));
        y = np.sin(np.radians(pos))
        ax.arrow(a * x, a * y, b * x, b * y, width=0.01, head_width=0.0, head_length=0, ls='--', fc='r', ec='r')

    # Panah pengukur
    pos = 180 - (180 * (max_val - arrow) / (max_val - min_val))
    pos_normalized = (arrow - min_val) / (max_val - min_val)
    angle_range = 180
    pos_degrees = angle_range * (1 - pos_normalized)

    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos_degrees)), 0.225 * np.sin(np.radians(pos_degrees)), \
             width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    return ax


# Parameter Streamlit (v1.25)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

# Penambahan logo
logo_path = 'logo_gaman.png'
st.sidebar.image(logo_path, use_column_width=True)

# Definisi kelompok kolom
categorical_columns = ['Jenis Kontrak',
                       'Jenis Kelamin',
                       'Gelar pendidikan tinggi',
                       'Kota kerja berbeda dengan kota tempat tinggal']

# Memuat model dari file pickle
model_path = './saved_model/'
with open('LightGBM_smote_tuned.pckl', 'rb') as f:
    model = pickle.load(f)

# Memuat dataframe dari dashboard
df_feature_importance = pd.read_csv('df_feature_importance_25_01.csv')
df_feature_importance.drop('Unnamed: 0', axis=1, inplace=True)
df_dashboard_final = pd.read_csv('df_dashboard_final_rev.csv')
df_dashboard_final.drop('Unnamed: 0', axis=1, inplace=True)

# Judul dashboard
st.title('Risiko kredit Usaha UMKM – Dashboard')

# Kotak di bagian kiri
st.sidebar.title('Pemilihan pelanggan/nasabah')
selected_client = st.sidebar.selectbox('Identitas Pelanggan/Nasabah :', df_dashboard_final['Nasabah ID'])
predict_button = st.sidebar.button('Prediksi')

# Mendapatkan indeks yang sesuai dengan ID pelanggan yang dipilih
index = df_dashboard_final[df_dashboard_final['Nasabah ID'] == selected_client].index[0]

# Menampilkan informasi dari pelanggan yang dipilih
client_info = df_dashboard_final[df_dashboard_final['Nasabah ID'] == selected_client]
st.subheader('Informasi pelanggan :')
client_info.index = client_info['Nasabah ID']
st.write(client_info[['Prediksi Kemampuan Bayar', 'Nilai Resiko Nasabah (dari skala 100)', 'Jenis Kontrak', 'Jenis Kelamin', 'Bagian angsuran dari total pendapatan', 'Jumlah Pendapatan Bulanan', 'Uang_Keluar_Payment_Gateway_3bulan', 'Nilai_Belanja_Online_3bulan', 'Biaya_Pulsa_3bulan', 'Biaya_Internet_dan_aplikasi_berbayar_3bulan']])

# Mendapatkan kategori prediksi kredit dari pelanggan yang dipilih
selected_client_cat = df_dashboard_final.loc[index, 'Prediksi Kemampuan Bayar']

# DataFrame yang berisi kategori prediksi kredit yang sama dengan pelanggan yang dipilih
df_customer = df_dashboard_final[df_dashboard_final['Prediksi Kemampuan Bayar'] == selected_client_cat].copy()

# Mendapatkan kategori prediksi kredit dari pelanggan yang dipilih
st.subheader('Tingkat Resiko Usaha UMKM:')
score = client_info['Nilai Resiko Nasabah (dari skala 100)'].values[0]  # Mengambil skor pelanggan
fig, ax = plt.subplots(figsize=(5, 3))
gauge(arrow=score, ax=ax)  # Memanggil fungsi gauge() dengan mengirimkan skor pelanggan
st.pyplot(fig)

# Bagian kanan layar
st.sidebar.title('Grafik')
univariate_options = [col for col in df_dashboard_final.columns if col not in ['Nasabah ID', 'Prediksi Kemampuan Bayar']]
bivariate_options = [col for col in df_dashboard_final.columns if col not in ['Nasabah ID', 'Prediksi Kemampuan Bayar']]

# Grafik univariat
univariate_feature = st.sidebar.selectbox('Variabel univariat :', univariate_options)
df_customer.replace([np.inf, -np.inf], 0, inplace=True)
st.subheader('Analisis univariat (populasi terbatas) :')
plt.figure()
plt.hist(df_customer[univariate_feature], color='skyblue', label='Populasi')
plt.xlabel(univariate_feature)
plt.axvline(client_info[univariate_feature].values[0], color='salmon', linestyle='--', label='Pelanggan/Nasabah yang terinvestigasi')
plt.legend()
st.pyplot(plt.gcf())

# Grafik bivariat
bivariate_feature1 = st.sidebar.selectbox('Variabel 1 (bivariat) :', bivariate_options)
bivariate_feature2 = st.sidebar.selectbox('Variabel 2 (bivariat) :', bivariate_options)
st.subheader('Analisis bivariat (populasi lengkap) :')
plt.figure()
sns.scatterplot(data=df_dashboard_final, x=bivariate_feature1, y=bivariate_feature2,
                c=df_dashboard_final['Nilai Resiko Nasabah'], cmap='viridis',
                alpha=0.5, label='Populasi')
sns.scatterplot(data=client_info, x=bivariate_feature1, y=bivariate_feature2,
                color='salmon', marker='o', s=100, label='Pelanggan/Nasabah yang terinvestigasi')
plt.xlabel(bivariate_feature1)
plt.ylabel(bivariate_feature2)
plt.legend()
st.pyplot(plt.gcf())

# Grafik kepentingan fitur global
df_sorted = df_feature_importance.sort_values('Features_importance_shapley', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Features_importance_shapley', y='Features', data=df_sorted, color='skyblue')
plt.xlabel('Tingkat Kepentingan SHAP')
plt.ylabel('Variabel')
st.subheader('Tingkat Kepentingan Variabel :')
st.pyplot(plt.gcf())
