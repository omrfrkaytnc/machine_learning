##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için, var olan müşterilerin
# gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel
# (hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
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

# GÖREV 2: CLTV Veri Yapısının Oluşturulması

# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması


###############################################################
# GÖREV 1: Veriyi Hazırlama
###############################################################

# 1. flo_data_20K.csv verisini okuyunuz. Dataframe’in kopyasını oluşturunuz.

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve
# replace_with_thresholds fonksiyonlarını tanımlayınız.

# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayınız.

# 4. Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.


import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None


# 1. flo_data_20K.csv verisini okuyunuz. Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv("2 -) CRM ANALYTIC/datasets/flo_data_20k.csv")
df = df_.copy()

# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve
# replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerlerinin integer olması gerekmektedir.
# Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.


#########    Outlier bulma ve baskılama fonksiyonlarını anlamak     #########
# istatistiksel modeller kullanacağımız için değişkenlerin dağılımı önemli,
# gerekiyorsa bu değişkenleri silmek yerine baskılayarak müdahale edebiliriz.


df.describe().T

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)

outlier_thresholds(df, "order_num_total_ever_online")
# (-27.5, 48.5)


# bizim veri setimiz low_limitin altında değere sahip olmadığı için sorun yok.


# 3. "order_num_total_ever_online","order_num_total_ever_offline",
# "customer_value_total_ever_offline","customer_value_total_ever_online"
# değişkenlerinin aykırı değerleri varsa baskılayınız.

columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
           "customer_value_total_ever_offline","customer_value_total_ever_online"]

outlier_thresholds(df, columns)

for col in columns:
    replace_with_thresholds(df, col)

df.describe().T
df_.describe().T

# 4. Omnichannel müşterilerin hem online'dan hem de offline platformlardan
# alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]


# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df.info()
###############################################################
# GÖREV 2: CLTV Veri Yapısının Oluşturulması
###############################################################

# CLTV - Müşterinin şirketle ilişkisi süresince şirkete kazandıracağı parasal değer
# harcama miktarı, alışveriş sıklığı, profit margin vb değişkenleri göz önüne alır.


# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını
# analiz tarihi olarak alınız.
df["last_order_date"].max() # 2021-05-30
analysis_date = dt.datetime(2021,6,1)


# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve
# monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.

## Monetary değeri satın alma başına ortalama değer olarak,
## recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]

# cltv recency - ilk satın alma ile son satın alma arasındaki fark, weekly -->7
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7

#Tenure - müşterinin yaşı, weekly -->7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype('timedelta64[D]'))/7


cltv_df["frequency"] = df["order_num_total"]


## Monetary - satın alma başına ortalama değer
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

cltv_df.head()



###############################################################
# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, 6 aylık CLTV'nin hesaplanması
###############################################################

# 1. BG/NBD modelini kurunuz.

# expected number of transaction'a odaklanır - frekans
# bgf - model nesnesi oluşturur, fit ederek modeli nihai haline getiririz.


bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])


# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz
# ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"] = bgf.predict(4*3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

cltv_df.head()

# 3. ayda en çok satın alım gerçekleştirecek 10 kişiyi incelemek istersek;
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]



# 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz
# ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])
cltv_df.head()



# 2.  Gamma-Gamma modelini fit ediniz.

#müşterinin işlem başına ortalama ne kadar kar getirebileceğini tahmin eder.

# Müşterilerin ortalama bırakacakları değeri tahminleyip
# exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()


# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

cltv_df.head()


# CLTV değeri en yüksek 20 kişiyi gözlemleyiniz.

cltv_df.sort_values("cltv",ascending=False)[:20]

cltv_df.sort_values("cltv",ascending=False)[:20]["customer_id"]



###############################################################
# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
###############################################################

# 1. 6 aylık standartlaştırılmış CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız
# ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile atayınız.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

(cltv_df.groupby("cltv_segment").agg({"sum", "mean", "max", "std"}))
(cltv_df.groupby("cltv_segment").agg({"mean", "max"}))

# CLTV skorlarına göre müşterileri 4 gruba ayırmak mantıklı mıdır?
# Daha az mı ya da daha çok mu olmalıdır. Yorumlayınız.


# 3. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa
# 6 aylık aksiyon önerilerinde bulununuz




###############################################################
# BONUS: Tüm süreci fonksiyonlaştırınız.
###############################################################

def create_cltv_df(dataframe):

    # Veriyi Hazırlama
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # CLTV veri yapısının oluşturulması
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # # Gamma-Gamma Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # Cltv tahmini
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentleme
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)


cltv_df.head(10)


