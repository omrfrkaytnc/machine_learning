
###############################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu
# (Customer Segmentation with Unsupervised Learning)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################

# Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering )
# müşteriler kümelere ayrılıp ve davranışları gözlemlenmek istenmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında
# OmniChannel(hem online hem offline) olarak yapan müşterilerin
# geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# 20.000 gözlem,
# 12 değişken

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı
#                 (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv.csv verisini okuyunuz.
           # 2. Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
#          Tenure(Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.

# GÖREV 2: K-Means ile Müşteri Segmentasyonu
           # 1. Değişkenleri standartlaştırınız.
           # 2. Optimum küme sayısını belirleyiniz.
           # 3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
           # 4. Herbir segmenti istatistiksel olarak inceleyeniz.

# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
           # 1. Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
           # 2. Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
           # 3. Herbir segmenti istatistiksel olarak inceleyeniz.


###############################################################
# GÖREV 1: Veri setini okutunuz ve müşterileri segmentlerken kullanıcağınız değişkenleri seçiniz.
###############################################################

import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np
import warnings
warnings.simplefilter(action="ignore")
import matplotlib
matplotlib.use("Qt5Agg")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)


df_ = pd.read_csv("6 -)  MACHINE LEARNING/datasets/flo_data_20k.csv")
df = df_.copy()

#veriye ilk bakış
df.head()
df.shape
# (19945, 12)

df.isnull().sum()
# eksik değer yok

df.info()

df.describe().T


# tarih değişkenine çevirme
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

df["last_order_date"].max()
#2021-05-30

analysis_date = dt.datetime(2021,6,1)


#recency: son alışverişten bu yana geçen süre
df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')


#tenure: müşterinin yaşı
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).astype('timedelta64[D]')

df.head()

# model df için sayısal değişkenleri seçelim
# (feature eng. kısmındaki yaklaşımı da kullanabilirdik)

model_df = df[["order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]

model_df.head()
model_df.info()


###############################################################
# GÖREV 2: K-Means ile Müşteri Segmentasyonu
###############################################################

# 1. Değişkenleri standartlaştırınız.
#SKEWNESS
def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show(block=True)


# Normal dağılımın sağlanması için Log transformation uygulanması

model_df['order_num_total_ever_online']=np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline']=np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline']=np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online']=np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency']=np.log1p(model_df['recency'])
model_df['tenure']=np.log1p(model_df['tenure'])
model_df.head()


# Scaling
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()


# 2. Optimum küme sayısını belirleyiniz.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show(block=True)


# 3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments = k_means.labels_
segments

final_df = df[["master_id",
               "order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]
final_df["segment"] = segments
final_df.head()

final_df.describe().T

#segmentler 0-6, biz 1-7 arası yapmak istersek

final_df["segment"] = final_df["segment"] + 1

# 4. Herbir segmenti istatistiksel olarak inceleyiniz.

final_df.groupby("segment").agg(["mean","min","max", "count"])


# final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
#                                   "order_num_total_ever_offline":["mean","min","max"],
#                                   "customer_value_total_ever_offline":["mean","min","max"],
#                                   "customer_value_total_ever_online":["mean","min","max"],
#                                   "recency":["mean","min","max"],
#                                   "tenure":["mean","min","max","count"]})



###############################################################
# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
###############################################################

# 1. Görev 2'de standarlaştırdığınız dataframe'i kullanarak
# optimum küme sayısını belirleyiniz.

model_df.info()
hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show(block=True)


# 2. Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)

final_df_h = df[["master_id","order_num_total_ever_online",
               "order_num_total_ever_offline",
               "customer_value_total_ever_offline",
               "customer_value_total_ever_online",
               "recency",
               "tenure"]]
final_df_h["segment"] = segments
final_df_h.head()

# 3. Herbir segmenti istatistiksel olarak inceleyeniz.

final_df_h.groupby("segment").agg(["mean","min","max", "count"])
