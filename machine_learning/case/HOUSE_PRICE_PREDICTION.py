
Ev Fiyat Tahmin Modeli

# Görev
# Elimizdeki veri seti üzerinden minimum hata ile ev fiyatlarını tahmin eden bir makine öğrenmesi modeli geliştiriniz ve kaggle yarışmasına tahminlerinizi yükleyiniz.

# İş Problemi

# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak,
# farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi gerçekleştirilmek istenmektedir.

"""Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor.

Kaggle üzerinde bir yarışması da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz.
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation

Veri seti bir kaggle yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır.

Test veri setinde ev fiyatları boş bırakılmış olup, bu değerleri sizin tahmin etmeniz beklenmektedir.
"""

# Toplam Gözlem: 2919
# Sayısal Değişken: 38
# Kategorik Değişken: 43

"""SalePrice:       the property's sale price in dollars. This is the target variable that you're trying to predict.

mülkün dolar cinsinden satış fiyatı. Bu, tahmin etmeye çalıştığınız hedef değişkendir.

MSSubClass:      The building class                              ---- Yapı sınıfı

MSZoning:        The general zoning classification               ---- Genel imar sınıflandırması

LotFrontage:     Linear feet of street connected to property     ---- Mülkiyete bağlı caddenin doğrusal ayakları

LotArea:         Lot size in square feet                         ---- Metrekare cinsinden lot büyüklüğü

Street:          Type of road access                             ---- Yol erişim türü

Alley:           Type of alley access                            ---- Sokak erişim türü

LotShape:        General shape of property                       ---- Mülkün genel şekli

LandContour:     Flatness of the property                        ---- Tesisin düzlüğü

Utilities:       Type of utilities available                     ---- Mevcut yardımcı programların türü

LotConfig:       Lot configuration                               ---- Lot yapılandırması

LandSlope:       Slope of property                               ----  Mülkün eğimi

Neighborhood:    Physical locations within Ames city limits      ---- Ames şehir sınırları içindeki fiziksel konumlar

Condition1:      Proximity to main road or railroad              ---- Ana yola veya demiryoluna yakınlık

Condition2:      Proximity to main road or railroad (if a second is present) ---- Ana yola veya demiryoluna yakınlık (bir saniye varsa)

BldgType:        Type of dwelling                                ---- Konut tipi

HouseStyle:      Style of dwelling                               ---- Konut tarzı

OverallQual:     Overall material and finish quality             ---- Genel malzeme ve kaplama kalitesi

OverallCond:     Overall condition rating                        ---- Genel durum değerlendirmesi

YearBuilt:       Original construction date                      ---- Orijinal yapım tarihi

YearRemodAdd:    Remodel date                                    ---- Tadilat tarihi

RoofStyle:       Type of roof                                    ---- Çatı tipi

RoofMatl:        Roof material                                   ---- Çatı malzemesi

Exterior1st:     Exterior covering on house                      ---- Evde dış kaplama

Exterior2nd:     Exterior covering on house (if more than one material) ---- Ev dış cephe kaplaması (birden fazla malzeme varsa)

MasVnrType:      Masonry veneer type                             ---- Kagir kaplama tipi

MasVnrArea:      Masonry veneer area in square feet              ---- Metrekare cinsinden kagir kaplama alanı

ExterQual:       Exterior material quality                       ---- Dış malzeme kalitesi

ExterCond:       Present condition of the material on the exterior     ---- Dış cephedeki malzemenin mevcut durumu

Foundation:      Type of foundation                              ---- Vakıf türü

BsmtQual:        Height of the basement                          ---- Bodrum katının yüksekliği

BsmtCond:        General condition of the basement               ---- Bodrum katının genel durumu

BsmtExposure:    Walkout or garden level basement walls          ---- Walkout veya bahçe seviyesinde bodrum duvarları

BsmtFinType1:    Quality of basement finished area               ---- Bodrum bitmiş alanın kalitesi

BsmtFinSF1:      Type 1 finished square feet                     ---- Tip 1 bitmiş metrekare

BsmtFinType2:    Quality of second finished area (if present)    ---- İkinci bitmiş alanın kalitesi (varsa)

BsmtFinSF2:      Type 2 finished square feet                     ---- Tip 2 bitmiş metrekare

BsmtUnfSF:       Unfinished square feet of basement area         ---- Bodrum alanının bitmemiş metrekaresi

TotalBsmtSF:     Total square feet of basement area              ---- Bodrum alanının toplam metrekaresi

Heating:         Type of heating                                 ---- Isıtma tipi

HeatingQC:       Heating quality and condition                   ---- Isıtma kalitesi ve durumu

CentralAir:      Central air conditioning                        ---- Merkezi klima

Electrical:      Electrical system                               ---- Elektrik sistemi

1stFlrSF:        First Floor square feet                         ---- Birinci Kat metrekare

2ndFlrSF:        Second floor square feet                        ---- İkinci kat metrekare

LowQualFinSF:    Low quality finished square feet (all floors)   ---- Düşük kaliteli bitmiş metrekare (tüm katlar)

GrLivArea:       Above grade (ground) living area square feet    ---- Sınıf üstü (zemin) yaşam alanı metrekare

BsmtFullBath:    Basement full bathrooms                         ---- Bodrum katı tam donanımlı banyolar

BsmtHalfBath:    Basement half bathrooms                         ---- Bodrum yarısı banyoları

FullBath:        Full bathrooms above grade                      ---- Sınıfın üzerinde tam banyolar

HalfBath:        Half baths above grade                          ---- Derecenin üzerinde yarım banyolar

Bedroom:         Number of bedrooms above basement level         ---- Bodrum katının üzerindeki yatak odası sayısı

Kitchen:         Number of kitchens                              ---- Mutfak sayısı

KitchenQual:     Kitchen quality                                 ---- Mutfak kalitesi

TotRmsAbvGrd:    Total rooms above grade (does not include bathrooms)   ---- Derecenin üzerindeki toplam oda sayısı (banyo dahil değildir)

Functional:      Home functionality rating                       ---- Ev işlevselliği derecelendirmesi

Fireplaces:      Number of fireplaces                            ---- Şömine sayısı

FireplaceQu:     Fireplace quality                               ---- Şömine kalitesi

GarageType:      Garage location                                 ---- Garaj konumu

GarageYrBlt:     Year garage was built                           ---- Garajın inşa edildiği yıl

GarageFinish:    Interior finish of the garage                   ---- Garajın iç kaplaması

GarageCars:      Size of garage in car capacity                  ---- Araç kapasitesindeki garajın büyüklüğü

GarageArea:      Size of garage in square feet                   ---- Metrekare cinsinden garajın büyüklüğü

GarageQual:      Garage quality                                  ---- Garaj kalitesi

GarageCond:      Garage condition                                ---- Garaj durumu

PavedDrive:      Paved driveway                                  ---- Asfalt garaj yolu

 WoodDeckSF:      Wood deck area in square feet                   ---- Metrekare ahşap güverte alanı

OpenPorchSF:     Open porch area in square feet                  ---- Metrekare cinsinden açık sundurma alanı

EnclosedPorch:   Enclosed porch area in square feet              ---- Metrekare cinsinden kapalı sundurma alanı

3SsnPorch:       Three season porch area in square feet          ---- Metrekare üç mevsim sundurma alanı

ScreenPorch:     Screen porch area in square feet                ---- Metrekare cinsinden ekran sundurma alanı

PoolArea:        Pool area in square feet                        ---- Metrekare cinsinden havuz alanı

PoolQC:          Pool quality                                    ---- Havuz kalitesi

Fence:           Fence quality                                   ---- Çit kalitesi

MiscFeature:     Miscellaneous feature not covered in other categories    ---- Diğer kategorilerde yer almayan çeşitli özellikler

MiscVal:         Value of miscellaneous feature                 ---- Çeşitli özelliklerin Value

MoSold:          Month Sold                                      ---- Satılan Ay

YrSold:          Year Sold                                       ---- Satılan Yıl

SaleType:        Type of sale                                    ---- Satış türü

SaleCondition:   Condition of sale                               ---- Satış durumu
"""

# Gerekli Kütüphane ve Fonksiyonları indirelim

!pip install catboost

!pip install xgboost
# pip install xgboost --default-timeout=60

!pip install lightgbm

# 1. GEREKLILIKLER

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

# data frame in sınırlamalarını kaldırıyoruz

pd.set_option('display.max_columns', None)  # bütün sütunları göster
pd.set_option('display.max_rows', None)     # bütün satırları göster
pd.set_option('display.width', 500)  # sütunlar max 500 tane gösterilsin
pd.set_option('display.expand_frame_repr', False)  # çıktının tek bir satırda olmasını sağlar

pd.set_option('display.float_format', lambda x: '%.3f' % x)  # virgülden sonra 3 basamak gösterir

"""GÖREV 1 : Veri setine EDA işlemlerini uygulayınız.

1. Genel Resim
2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
4. Hedef Değişken Analizi (Analysis of Target Variable)
5. Korelasyon Analizi (Analysis of Correlation)
"""



"""
Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz."""

# train ve test setlerinin bir araya getirilmesi.
train = pd.read_csv("6 -)  MACHINE LEARNING/datasets/house_price_train.csv")
test = pd.read_csv("6 -)  MACHINE LEARNING/datasets/house_price_test.csv")


df = train.append(test,ignore_index=False).reset_index()  # ignore_index=False parametresi, orijinal DataFrame'lerin indekslerini korumak için kullanılır.
# Eğer True olsaydı, yeni birleştirilmiş DataFrame için yeni indeksler oluşturulacaktı

df.head()

df.tail()

df = df.drop("index", axis=1) # axis=1 parametresi, silinecek şeyin bir sütun olduğunu belirtir (axis=0 satırlar için kullanılır)
# Bu satır, df DataFrame'inden "index" adlı sütunu siler (drop fonksiyonu ile).

# Kısaca bu adım, bir önceki adımda eklenen eski indeks sütununu kaldırır, çünkü genellikle analizde gerekli değildir.

df.head()

"""1. Genel Resim"""

df.isnull().sum()
# MSZoning            4 ---- Genel imar sınıflandırması
# LotFrontage       486
# Alley            2721
# Utilities           2
# Exterior1st         1
# Exterior2nd         1
# MasVnrType         24
# MasVnrArea         23
# BsmtQual           81
# BsmtCond           82
# BsmtExposure       82
# BsmtFinType1       79
# BsmtFinSF1          1
# BsmtFinType2       80
# BsmtFinSF2          1
# BsmtUnfSF           1
# TotalBsmtSF         1
# Electrical          1
# BsmtFullBath        2
# BsmtHalfBath        2
# KitchenQual         1
# Functional          2
# FireplaceQu      1420
# GarageType        157
# GarageYrBlt       159
# GarageFinish      159
# GarageCars          1
# GarageArea          1
# GarageQual        159
# GarageCond        159
# PoolQC           2909
# Fence            2348
# MiscFeature      2814
# SaleType            1
# SalePrice        1459

# Dataframe in genel resmine tek seferde bakıyoruz

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

"""NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI"""

# bu fonksiyon bize gizli numerikleri ve gizli kategorikleri gösterecek

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

# df veri setimizi grab_col_names fonksiyonundan geçiriyoruz
cat_cols, cat_but_car, num_cols = grab_col_names(df)

"""2. Kategorik Değişken Analizi (Analysis of Categorical Variables)"""

# KATEGORİK DEĞİŞKENLERİN ANALİZİ
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

"""3. Sayısal Değişken Analizi (Analysis of Numerical Variables)"""

# NUMERİK DEĞİŞKENLERİN ANALİZİ
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col, True)

"""4. Hedef Değişken Analizi (Analysis of Target Variable)"""

# KATEGORİK DEĞİŞKENLERİN TARGET'A GÖRE ANALİZİ
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"SalePrice",col)

# Bağımlı değişkenin incelenmesi histogram grafiği
# bağımlı değişkenin normal dağılmasını bekleriz ancak bir sağa çarpıklık söz konusu solda yığılma var
# bu dağılımı normalleştirmek için bazı işlemler yapabilmekteyiz
df["SalePrice"].hist(bins=100)
plt.show()

# Bağımlı değişkenin logaritmasının incelenmesi
# logartimik dönüşümü bağımlı değişkene uyguluyoruz, bu dönüşüm dağılımı biraz daha normalleştiriyor.
# model kurarken bağımlı değişkenin logaritmik dönüştürülmüş haliyle işlem yapabiliriz
np.log1p(df['SalePrice']).hist(bins=50)
plt.show()

"""5. Korelasyon Analizi (Analysis of Correlation)"""

# değişkenler arasındaki ilişkiyi incelemek için korelasyon analizine bakalım
# korelasyon iki değişken arasındaki ilişkinin yönünü ve derecesini gösterir
# -1 ile +1 arasında değişir ve 0 herhangi bir ilişki olmadığını gösterir
# -1 e yaklaştıkça negatif güçlü ilişki, +1 e yaklaştıkça pozitif güçlü ilişki olduğunu gösterir

corr = df[num_cols].corr()
corr

# Korelasyonların gösterilmesi
# renk kırmızıya doğru kaydıkça negatif güçlü ilişki artmaktadır,
# renk koyu maviye doğru kaydıkça da pozitif güçlü ilişki artmaktadır
sns.set(rc={'figure.figsize': (15, 15)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# KORELASYON ANALİZİ FARKLI BİR GÖSTERİM (tipi numerik olanlar ile)
df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# KORELASYON ANALİZİ FARKLI BİR GÖSTERİM (tipi kategorik olanlar ile)
df[cat_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[cat_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# bağımlı değişken ile bağımsız değişken arasında güçlü ilişki olsun isteriz. Çünkü bağımsız değişken bağımlı değişkeni etkileyen onun hakkında bilgi veren ve onu açıklayan değişkenlerdir.
# Ancak bağımsız değişkenler arasında çok fazla güçlü ilişki olmasını istemeyiz çünkü birbirinden etkilenmesini istemeyiz
# etkilenmesi durumu da bize çoklu doğrusal bağlantı sorununa yol açar. Bunu istemeyiz ancak bu regresyon modeli için geçerlidir. diğer durumlarda bu bağlantı gözardı edilebilmektedir.

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=False)


# yüksek korelasyona sahip değişkenler
# ['1stFlrSF', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea', 'SalePrice']

"""Görev 2 : Feature Engineering

Aykırı Değer Analizi
"""

# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# baskılama aykırı değerlerin en alt değerlere ve en üst değerlere göre sabitlenmesi durumudur

# thresholdumuzu veri setimize göre gözlem sayısına göre, değişkenlerin yapısına göre kendi know-how ımıza göre belirleyebiliriz
# genel geçer %75 e %25 şeklinde alınandır. ancak çok fazla bilgi kaybetmemek için bu değerleri büyütmek mümkündür
# fazlaca baskılama yapmak çoğu zaman bilgi kaybına ve değişkenlerin arasındaki ilişkinin kaybolmasına neden olabilmektedir

# Aykırı değer kontrolü

# bu eşik değerlerlere göre aykırı değerler var mı değişkenlerde, varsa hangilerinde var kontrol edeceğiz
# bir değişkenin aykırı değerlerini bool olarak sorgulatacağız
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "SalePrice":
      print(col, check_outlier(df, col))

# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df,col)

# tekrar bakalım aykırı değer kalmış mı
for col in num_cols:
    print(col, check_outlier(df, col))

"""Eksik Değer Analizi"""

msno.bar(df)
plt.show()
# "Alley", "PoolQC", "Fence", "MiscFeature"  çok fazla eksik gözlem var
# sokak erişim türü, havuz kalitesi, çit kalitesi, Diğer kategorilerde yer almayan çeşitli özellikler

# bu fonksiyonla elimize bir veri geldiğinde bu veriyi hızlı bir şekilde eksikliklerin frekansı nedir
# hangi değişkenlerde eksiklik var ve bu eksikliklerin oranı nedir bilgisini göreceğiz

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

#
df["Alley"].value_counts()

#
df["BsmtQual"].value_counts()

# Bazı değişkenlerdeki boş değerler evin o özelliğe sahip olmadığını ifade etmektedir,
# bu kanıya data seti iyice inceleyerek ve data setin ve değişkenlerin dinamiklerine bakarak karar vermeliyiz
# örneğin PoolQC bir gözlemde boş ise o evde havuz olmadığını belirtmektedir.
no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

# Kolonlardaki boşlukların "No" ifadesi ile doldurulması
# burada değişkenler kendi nezdinde incelemeli hepsine medyan ya da mod ya da ortalama uygulamak yerine değişken bazında
# uygun metrik ile doldurmak daha uygun olacaktır
for col in no_cols:
    df[col].fillna("No",inplace=True)

missing_values_table(df)

"""Eksik veri problemi nasıl çözülür
###########################################

Silme : eksik verilerin değişkenlerinin silinmesi
'''
dropna diyerek silebiliriz ama bu durumda gözlem sayısı azalacaktır. gözlem sayısı çok fazla ve
eksik gözlem sayısı az ise eksik değerler silinebilir ancak gözlem sayısının az olduğu ya da eksik değerlerin
fazlaca bulunduğu verilerde silme işlemi yapmak ciddi oranda bir veri kaybına yol açacaktır
'''


Değer atama yöntemleri : ortalama mod medyan gibi basit atama yöntemleri
'''
değişkenlerdeki eksiklikleri medyanı ya da ortalamasıyla doldurabileceğimiz gibi her hangi bir sabit değerle de doldurabiliriz
'''

Tahmine dayalı yöntemler : ML ya da istatistiksel bazı yöntemlerle tahminlere göre değer atama
################################################################################

biz şimdi mode medyan yöntemleri ile atamaya geçiş yapacağız

nümerik değişkenlerin eksiklerinin tamamlanması


Bu fonsksiyonun ön tanımlı değeri medyandır. bunu daha sonra num_method="XXX" girerek değiştirebiliriz

eksik değerlerin median veya mean ile doldurulmasını sağlar

categorik değişkenler için eşik 20 belirlenmiştir bu değişkenin sahip olabileceği maksimum

eşsiz sınıf sayısını ifade eder. Varsayılan target değer de "SalePrice" şeklindedir.

Fonksiyon önce veri kümesindeki eksik değerlere sahip değişkenleri tanımlar ve bunları variables_with_na adlı bir listede saklar. Daha sonra hedef değişkeni temp_target adlı geçici bir değişkende saklar.

Ardından fonksiyon, veri kümesindeki her sütuna bir lambda işlevi uygulamak için Apply() yöntemini kullanır.

Lambda işlevi, her sütunun veri tipini ve benzersiz değerlerinin sayısını kontrol eder ve eksik değerleri şu şekilde doldurur:

Veri türü "O" (yani nesne) ise ve benzersiz değerlerin sayısı cat_length=20 küçük veya ona eşitse, mod (yani en sık kullanılan değer) ile eksik değerler atanır.

Eğer num_method "mean" ise, "O" dışında veri tipine sahip sütunlardaki eksik değerler ortalama değerle ilişkilendirilir.

Eğer num_method "medyan" ise, "O" dışında veri tipine sahip sütunlardaki eksik değerler medyan değerle hesaplanır.

Son olarak, işlev hedef değişkeni geri yükler ve atamadan önce ve sonra her sütundaki eksik değerlerin sayısını yazdırır.

Daha sonra değiştirilen veri kümesini döndürür.
"""

# Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar

def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)

# tekrar kontrol edelim
missing_values_table(df)
# ve hiç eksiklik kalmadı, SalePrice hariç

# eksik değerlerin olduğu değişkenlere bakalım
[col for col in df.columns if df[col].isnull().sum() > 0]
# Out[23]: ['SalePrice'] sale hariç kalmadı

"""Rare analizi yapınız ve rare encoder uygulayınız."""

# buradaki kategorik değişkenleri seçmemiz lazım grab col names i çağıracağız
# kategorik değişkenleri getiriyoruz
# neden bunu yapıyoruz
# gereksiz sayıda kategori olmasın ve benzer kategorileri bir araya getirelim ya da işe yaramayanları çıkartalım
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# şimdi bu kategorik değişkenlerimizi ve sınıflarını, sınıfların azlık çokluk durumlarına göre analiz edelim

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
######################################################################
# şimdi burda yani bir fonksiyona ihtiyacımız var ve bu fonksiyonla öyle bir işlem yapmamız lazım ki
# kategorik değişkenlerin sınıflarını ve bu sınıfların oranlarını getirsin, plot=true dersek grafikler de gelir

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# bunu neden yaptık

# rare az gözlemlenen nadir demektir
# az gözlemlenen değişkenler one hot encoding işlemi yaptığımızda sütununda fazla bilgi barındırmayacaktır
# bu nedenle modellemede karşılık bulamayacaklardır
# bu nedenle one hot encoderdan geçirip değişken haline getirdiğimiz değerlerin de ölçüm kalitesinin olmasını
# ve bağımlı değişkene çeşitli olası etkilerini göz önünde bulundurmak isteriz.
# bu nedenle gereksiz değişkenlerden uzaklaşmak kurtulmak için rare encoder ı kullanabiliriz
# toparlamak gerekirse veri setindeki bir kategorik değişkenin sınıflarındaki az değerlerden kurtulmak için
# bir eşik değeri belirleriz ve bu belirlediğimiz belirli bir eşik değerine göre altta kalan sınıfları
# toparlarız bir araya getirip bunlara rare deriz yani bir bakıma dışarda bırakırız

# Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.

# şimdi bu rare kategorisine alacağımız sınıfların değişkenlerin bağımlı değişkene etkileri nedir ve
# arasındaki ilişki nasıldır bunu analiz edeceğiz

# bunu neden yapıyoruz? gereksiz sayıda kategori olmasın, birbirlerine benzeyen yerleri, değişkenleri veya kategorileri olabildiğince bir araya getirelim
# bilgi vermeyen sınıfları bir araya getirelim bilgi vermeyen sınıflardan kurtulalım ya da onları başka bir kategoriye dahil edelim

# bu fonskiyonla sınıfların frekansları oranları ve target yani SalePrice açıdından ortalamaları gelecek

# Kategorik kolonların dağılımının incelenmesi
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)

# Rare encoder'ın yazılması.

# rare yüzdemizi belirleyeceğiz bu oranın altında kalan kategorik değişken sınıflarını bir araya getirecek.
# rare encoderımız veri setindeki seyrek sınıflı kategorik değişkenlerin seyrek sınıflarını toplayıp bir araya getirerek
# bunlara rare isimlendirmesi yapmaktadır

# Nadir sınıfların tespit edilmesi
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


rare_encoder(df, 0.01)

# dff diye kaydediyoruz bütün kategorik değişkenlerin sınıflarını rare encoderdan geçirdikten sonra

# böylelikle df te bir sorun olduğunda her şeyi baştan çalıştırmak yeine burdan yeni oluşturduğumuz dff ile devam edebiliriz

dff = rare_encoder(df, 0.01)

# Rare altında topladıktan sonra tekrar bakalım rare analiz çıktımıza
rare_analyser(dff, "SalePrice", cat_cols)
dff.head()

"""Yeni değişkenler oluşturunuz ve oluşturduğunuz yeni değişkenlerin başına 'NEW' ekleyiniz."""

df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]

df["NEW_Garage*GrLiv"] = (df["GarageArea"] * df["GrLivArea"])

df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1) # 42


# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"] # 32

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2 # 56

# Porch Area
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF # 93

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF # 156

df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF # 35


# Lot Ratio
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea # 64

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea # 57

df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea # 69

# MasVnrArea
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea # 36

# Dif Area
df["NEW_DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF) # 73


df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"] # 61


df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt # 31

df["NEW_HouseAge"] = df.YrSold - df.YearBuilt # 73

df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd # 40

df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt # 17

df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd) # 30

df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt # 48

# kolonlar üzerinden yeni feature lar ürettik ve eskilerine gerek kalmadı bu yüzden bunlara ihtiyacımız yok ve data frame den düşüreceğiz
drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

# drop_list'teki değişkenlerin düşürülmesi
df.drop(drop_list, axis=1, inplace=True)

df.shape

"""Label Encoding & One-Hot Encoding işlemlerini uygulayınız."""

# Değişkenlerin tiplerine göre ayrılması işlemi yeni değişkenlerden sonra
cat_cols, cat_but_car, num_cols = grab_col_names(df)

# label encoding / binary encoding işlemini 2 sınıflı kategorik değişkenlere uyguluyoruz
# yani nominal sınıflı kategorik değişkenlere böylelikle bu iki sınıfı 1-0 şeklinde encodelamış oluyoruz

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)

df.head()

# one-hot encoder ise ordinal sınıflı kategorik değişkenler için uyguluyoruz. sınıfları arasında fark olan
# değişkenleri sınıf sayısınca numaralandırıp kategorik değişken olarak df e gönderiyor

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

"""MODELLEME

GÖREV 3: Model kurma
"""

#  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

# Train verisi ile model kurup, model başarısını değerlendiriniz.
# bağımlı ve bağımsız değişkenleri seçiyoruz

# Sale price çarpık bir dağılıma sahipti, öncelikle log dönüşümü yapmadan modelleme kuracağız
# daha sonra da log dönüşümü yaparak model kuracağız ve rmse değerlerimizi log öncesi ve log sonrasına göre karşılaştıracağız
y = train_df['SalePrice'] # np.log1p(df['SalePrice'])  y= bağımlı değişken
X = train_df.drop(["Id", "SalePrice"], axis=1)        # X = Id hariç bağımsız değişkenler (90 değişkenle beraber)

# Train verisi ile model kurup, model başarısını değerlendiriniz.
# modelimizi kuruyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# kullanacağımız yöntemi import ettik
from lightgbm import LGBMRegressor

# bağımlı değişkenimiz sayısal ise regression, regressor algoritmalarını
# bağımlı değişkenimiz kategorikse classification algoritmalarını kullanıyoruz

# kullanacağımız yöntemleri içeren bir model tanımlı nesne kuruyoruz
# kapalı olan algoritmaları da açarak onları da modele sokabilirsiniz
models = [('LR', LinearRegression()),
          #("Ridge", Ridge()),
          #("Lasso", Lasso()),
          #("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

# daha sonra model nesnemizi döngü ile rmse değerini her bir yöntem için hesaplayacak şekilde
# fonksiyonel olarak çağırıyoruz

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 42501.7985 (LR)
# RMSE: 47557.3947 (KNN)
# RMSE: 38786.4984 (CART)
# RMSE: 28910.3004 (RF)
# RMSE: 25725.1132 (GBM)
# RMSE: 27971.7767 (XGBoost)
# RMSE: 28582.004 (LightGBM)
# RMSE: 25551.3003 (CatBoost)

df['SalePrice'].mean()

df['SalePrice'].std()

"""**Standart Sapma (Standard Deviation)**
Standart sapma, veri noktalarının ortalamadan ne kadar farklılık gösterdiğinin bir ölçüsüdür. Yani, veri setindeki değerlerin dağılımının ne kadar yaygın olduğunu gösterir. Standart sapma ne kadar büyükse, veri noktaları ortalamadan o kadar çok sapar ve dağılım o kadar geniştir.

**Ortalama ve Standart Sapma Arasındaki Fark ve Kıyaslama**
Ortalama, veri setinin merkezi eğilimini temsil ederken, standart sapma veri noktalarının bu merkeze olan ortalama uzaklığını temsil eder. Birlikte, veri setinin genel yapısını ve dağılımını anlamamıza yardımcı olurlar.

**Düşük Standart Sapma:** Eğer standart sapma değeri düşükse, bu veri noktalarının ortalamaya yakın olduğunu ve veri setinin oldukça homojen olduğunu gösterir. Yani, veri noktaları birbirine benzer ve tutarlıdır.

**Yüksek Standart Sapma:** Yüksek standart sapma, veri noktalarının ortalamadan büyük ölçüde sapmalar gösterdiğini ve veri setinin heterojen olduğunu gösterir. Veri noktaları arasında büyük farklılıklar olabilir ve veri seti daha değişkendir.

**Kıyaslama Yöntemi:** Ortalama ve standart sapmayı kıyaslamak için doğrudan bir "fark" hesaplaması genellikle yapılmaz. Bunun yerine, standart sapmanın büyüklüğünü ortalamaya göre değerlendiririz. Örneğin, ortalamaya oranla standart sapmanın büyük veya küçük olması, veri dağılımının yaygınlığı hakkında bilgi verir.
"""



"""BONUS : Log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz.

Not: Log'un tersini (inverse) almayı unutmayınız.
"""

# Log dönüşümünün gerçekleştirilmesi

# tekrardan Train ve Test verisini ayırıyoruz.
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

# Bağımlı değişkeni normal dağılıma yaklaştırarak model kuracağız

y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

# Verinin eğitim ve test verisi olarak bölünmesi
# log dönüşümlü hali ile model kuruyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# lgbm_tuned = LGBMRegressor(**lgbm_gs_best.best_params_).fit(X_train, y_train)

# bağımlı değişkendeki log dönüştürülmüş tahminlemelere bakıyoruz

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
y_pred
# Bağımlı değişkendeki gözlemlerin tahminlemiş halleri geliyor (log dönüştürülmüş halleri geldi tabi)
# gerçek değerlerle karşılaştırma yapabilmek için bu log dönüşümünün tekrar tersini (inverse) almamız gerekmektedir.

# daha sonra model nesnemizi döngü ile rmse değerini her bir yöntem için hesaplayacak şekilde
# fonksiyonel olarak çağırıyoruz

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# # LOG DÖNÜŞÜMÜ ÖNCESİ
# RMSE: 42501.7985 (LR)
# RMSE: 47557.3947 (KNN)
# RMSE: 38786.4984 (CART)
# RMSE: 28910.3004 (RF)
# RMSE: 25725.1132 (GBM)
# RMSE: 27971.7767 (XGBoost)
# RMSE: 28582.004 (LightGBM)
# RMSE: 25551.3003 (CatBoost)

# LOG DÖNÜŞÜMÜ SONRASI
# RMSE: 0.1547 (LR)
# RMSE: 0.2357 (KNN)
# RMSE: 0.2047 (CART)
# RMSE: 0.1419 (RF)
# RMSE: 0.1301 (GBM)
# RMSE: 0.1427 (XGBoost)
# RMSE: 0.1343 (LightGBM)
# RMSE: 0.1239 (CatBoost)

# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması (y_pred için)
new_y = np.expm1(y_pred)
new_y
# burada y_pred değerleri log dönüşümü yapılmış hedef değişken tahminlerini gösterirken
# new_y değeri y_pred in inverse uygulanmış yani log dönüşümünün tersinin yapılmış halinin tahmin sonuçlarını göstermektedir.
# bu iki değerlerin çıktılarını yani log dönüşümlü ve dönüşümsüz hallerini karşılaştırabilirsiniz

# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması (y_test için)
new_y_test = np.expm1(y_test)
new_y_test

# Inverse alınan yani log dönüşümü yapılan tahminlerin RMSE değeri
np.sqrt(mean_squared_error(new_y_test, new_y))

"""Log dönüşümü ve ardından yapılan inverse log dönüşümü (log dönüşümünün tersi), veri biliminde özellikle regresyon modellerinde sıkça karşılaşılan bir tekniktir. Bu tekniklerin kullanılmasının başlıca nedenleri şunlardır:

**Veri Dağılımını Düzeltmek:** Gerçek dünyada karşılaşılan birçok veri seti, normal dağılımdan sapmalar gösterir. Özellikle, hedef değişkenin sağa ya da sola çarpık olduğu durumlar, lineer regresyon gibi bazı algoritmaların varsayımlarını ihlal edebilir. Log dönüşümü, çarpık veriyi daha normal bir dağılıma dönüştürerek bu algoritmaların daha iyi performans göstermesine yardımcı olabilir.

**Hata Terimlerinin Varyansını Sabitlemek:** Regresyon modelleri için bir diğer önemli varsayım, hata terimlerinin sabit bir varyansa (homoscedasticity) sahip olmasıdır. Çarpık verilerde, büyük değerlere sahip gözlemler genellikle daha büyük hata terimlerine sahip olabilir. Log dönüşümü, bu varyansı sabitleyerek modelin daha tutarlı tahminler yapmasına olanak tanır.

**Çok Büyük veya Çok Küçük Değerlerle Başa Çıkmak:** Bazı durumlarda, hedef değişkende çok büyük veya çok küçük değerler olabilir. Bu tür değerler, modelin öğrenme sürecini olumsuz etkileyebilir. Log dönüşümü, değer aralığını sıkıştırarak bu sorunu hafifletebilir.

*Veriye log dönüşümü uygulandıktan sonra, modelin tahminleri de log dönüşümlü hedef değişken üzerinde yapılmış olur. Ancak, gerçek dünya uygulamalarında tahminlerin orijinal ölçeğe dönüştürülmesi gerekir. Bu nedenle, modelin çıkışındaki tahminlerin log dönüşümünün tersi alınarak orijinal ölçeğe dönüştürülmesi gerekir.* Bu işlem, np.expm1 fonksiyonu ile gerçekleştirilir. np.expm1(x) fonksiyonu, exp(x) - 1 hesaplamasını yapar ve bu, log1p dönüşümünün (yani log(1+x)) tersidir.

*Modelin performansını değerlendirirken, tahminlerin ve gerçek değerlerin orijinal ölçeğe dönüştürülmesi, elde edilen hata metriğinin (örneğin, RMSE) daha anlamlı ve yorumlanabilir olmasını sağlar. Çünkü son kullanıcılar veya karar vericiler, modelin çıktılarını ve performansını orijinal ölçekte anlamak isteyecektir.*

Hiperparametre optimizasyonlarını gerçekleştiriniz.
"""

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# rmse: 0.13433133803712316
# bu henüz hiç bir hiperparametre ayarlaması yapılmamış base modelin rmse sonucudur,
# aşağıda hiperparametre optimizasyonu yaptıktan sonra tekrar bir rmse değeri bakacağız ve bu değerle onu karşılaştır. Düşüş gözlemlenmeli

lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# rmse: 0.1296004503932677
# bu hiperparametre optimizasyonu yapılmış final modelin rmse değeridir

"""**ŞİMDİ BİR DE CATBOOST İLE MODEL KURALIM**"""

catboost_model = CatBoostRegressor(random_state=17)

rmse = np.mean(np.sqrt(-cross_val_score(catboost_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# rmse: 0.12285753813282337
# catboost base modelinin rmse sonucu

catboost_model.get_params()

"""CatBoostRegressor ve LGBMRegressor gibi farklı makine öğrenmesi kütüphanelerinin get_params() metodunun farklı çıktılar vermesi, bu kütüphanelerin implementasyonları ve varsayılan parametrelerin nasıl yönetildiğiyle ilgilidir. Her iki kütüphane de Python'da sınıflar aracılığıyla implemente edilmiştir ve get_params() metodu, bir sınıfın anlık (instance) özelliklerini (yani parametrelerini) bir sözlük olarak döndürür. Ancak, bir kütüphanenin get_params() metodu çağrıldığında hangi parametrelerin görüntüleneceği, o kütüphanenin nasıl tasarlandığına bağlıdır.

**CatBoost**
CatBoostRegressor'ın get_params() metodunun sadece {'loss_function': 'RMSE', 'random_state': 46} gibi sınırlı bir çıktı vermesinin nedeni, CatBoost'un yalnızca değiştirilmiş veya açıkça belirtilmiş parametreleri döndürmesidir. Yani, eğer bir parametre varsayılan değerini koruyorsa ve bu değer CatBoost tarafından içsel olarak yönetiliyorsa, bu parametre get_params() çıktısında görünmeyebilir. CatBoost, kullanıcının belirtmediği parametreler için genellikle içsel varsayılan değerleri kullanır ve bu yüzden get_params() çıktısı daha minimal olabilir.

**LightGBM**
Öte yandan, LGBMRegressor'un get_params() metodunun daha fazla parametre bilgisi vermesi, LightGBM'in varsayılan parametrelerini açık bir şekilde kullanıcının erişimine sunmasıyla ilgilidir. LightGBM, oluşturulduğu anda tüm varsayılan parametreleri açıkça belirler ve bunları get_params() çıktısında döndürür.

**Hiperparametre Ayarlama**
Bu farklılık, hiperparametre ayarlama kabiliyetinizi etkilemez. Her iki kütüphane de, modelinizi oluştururken veya modelinizi oluşturduktan sonra hiperparametreleri ayarlamanıza olanak tanır. CatBoostRegressor ve LGBMRegressor için hiperparametre optimizasyonu yapabilir ve model performansınızı iyileştirebilirsiniz. Örneğin, Grid Search, Random Search veya Bayesian Optimization gibi yöntemlerle en iyi parametre setini bulabilirsiniz.

CatBoost için hiperparametre ayarlama yaparken, dökümantasyonda belirtilen tüm parametreleri inceleyebilir ve ihtiyaçlarınıza göre bunları ayarlayabilirsiniz. Ayarladığınız parametreler, get_params() metodu çağrıldığında döndürülen sözlükte görünecektir.

Sonuç olarak, get_params() metodunun farklı çıktıları, kütüphanelerin tasarım farklılıklarından kaynaklanmaktadır ve hiperparametre ayarlama yeteneğinizi etkilemez. Her iki kütüphane de geniş bir hiperparametre setini destekler ve bu parametreler üzerinde optimizasyon yaparak model performansını artırabilirsiniz.






"""

# CatBoost'un resmi dokümantasyonu, kullanılabilir tüm parametreleri ve her birinin varsayılan değerlerini içerir.
#CatBoostRegressor ve diğer CatBoost modelleri için parametre referanslarına dokümantasyondan ulaşabilirsiniz.

# Python'un yerleşik help() fonksiyonunu kullanarak CatBoostRegressor sınıfının dokümantasyonuna erişebilir ve parametreleri hakkında bilgi alabilirsiniz.
# Bu yöntem, interaktif Python oturumları veya Jupyter Notebook'larda hızlı bir şekilde parametreleri gözden geçirmek için kullanışlıdır.
from catboost import CatBoostRegressor
help(CatBoostRegressor)

# Yukarıdaki dökümentasyona göre catboost için gelen parametre değerleri ve varsayılan değerleri aşağıya listelenmiştir
iterations=None
learning_rate=None
depth=None
l2_leaf_reg=None
model_size_reg=None
rsm=None
loss_function='RMSE'
border_count=None
feature_border_type=None
per_float_feature_quantization=None
input_borders=None
output_borders=None
fold_permutation_block=None
od_pval=None
od_wait=None
od_type=None
nan_mode=None
counter_calc_method=None
leaf_estimation_iterations=None
leaf_estimation_method=None
thread_count=None
random_seed=None
use_best_model=None
best_model_min_trees=None
verbose=None
silent=None
logging_level=None
metric_period=None
ctr_leaf_count_limit=None
store_all_simple_ctr=None
max_ctr_complexity=None
has_time=None
allow_const_label=None
target_border=None
one_hot_max_size=None
random_strength=None
random_score_type=None
name=None
ignored_features=None
train_dir=None
custom_metric=None
eval_metric=None
bagging_temperature=None
save_snapshot=None
snapshot_file=None
snapshot_interval=None
fold_len_multiplier=None
used_ram_limit=None
gpu_ram_part=None
pinned_memory_size=None
allow_writing_files=None
final_ctr_computation_mode=None
approx_on_full_history=None
boosting_type=None
simple_ctr=None
combinations_ctr=None
per_feature_ctr=None
ctr_description=None
ctr_target_border_count=None
task_type=None
device_config=None
devices=None
bootstrap_type=None
subsample=None
mvs_reg=None
sampling_frequency=None
sampling_unit=None
dev_score_calc_obj_block_size=None
dev_efb_max_buckets=None
sparse_features_conflict_fraction=None
max_depth=None
n_estimators=None
num_boost_round=None
num_trees=None
colsample_bylevel=None
random_state=None
reg_lambda=None
objective=None
eta=None
max_bin=None
gpu_cat_features_storage=None
data_partition=None
metadata=None
early_stopping_rounds=None
cat_features=None
grow_policy=None
min_data_in_leaf=None
min_child_samples=None
max_leaves=None
num_leaves=None
score_function=None
leaf_estimation_backtracking=None
ctr_history_unit=None
monotone_constraints=None
feature_weights=None
penalties_coefficient=None
first_feature_use_penalties=None
per_object_feature_penalties=None
model_shrink_rate=None
model_shrink_mode=None
langevin=None
diffusion_temperature=None
posterior_sampling=None
boost_from_average=None
text_features=None
tokenizers=None
dictionaries=None
feature_calcers=None
text_processing=None
embedding_features=None
eval_fraction=None
fixed_binary_splits=None

# Yukarıdan çektiğimiz bazı parametrelere göre yeni bir catboost hiperparametre ayarlamaları yapalım
catboost_params = {
    "n_estimators": [100, 500],
    "learning_rate": [0.01, 0.1],
    "max_depth": [2, 3],
    "random_state": [17],  # Modelin tekrarlanabilirliğini sağlamak için
    # Daha fazla parametre eklenebilir, örneğin:
    "l2_leaf_reg": [1, 3, 5],  # Düzenlileştirme terimi
    # "border_count": [32, 64, 128],  # Sayısal özellikler için sınır sayısı (bölme noktaları)
    # "auto_class_weights": ['Balanced', None],  # Sınıflar arası dengesizliği düzeltmek için
    # "bootstrap_type": ['Bayesian', 'Bernoulli', 'MVS'],  # Örnekleme yöntemi
    # Özelleştirilebilir başka parametreler...
}

catboost_gs_best = GridSearchCV(catboost_model,
                            catboost_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

# Yeni bir CatBoostRegressor nesnesi oluşturun ve oluşturulan yeni modeli eğitin
final_model2 = catboost_model.set_params(**catboost_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model2, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# rmse: 0.12686092071271032
# bu hiperparametre optimizasyonu yapılmış final modelin rmse değeridir
# skorun azalmadığını aksine arttığını görüyoruz burada parametreler azaltılıp değerleriyle oynanarak optimal rmse değerine gidilmeye çalışılabilinir.

"""Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz."""

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:50])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

#lightgbm modeli ile plot importance grafiğini çıkartıyoruz
model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)

#catboost modeli ile plot importance grafiğini çıkartıyoruz
model2 = CatBoostRegressor()
model2.fit(X, y)

plot_importance(model2, X)

# buradan catboost çıktılarıyla lightgbm çıktılarının görselleri karşılaştırılarak hangi modelin daha uygun olduğuna ve değişkenlere yanıt verdiğine dair bir yorumlama yapabiliriz

"""test dataframe indeki boş olan salePrice değişkenlerini tahminleyiniz

Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturunuz. (Id, SalePrice)
"""

# şimdi test veri setinde hedef değişkenimizin tahminlemesini yapıyoruz
model = LGBMRegressor()
model.fit(X, y)

# test veri setinden Id ve SalePrice sütunlarını çıkarıp ve geriye kalan sütunları modelin tahmin yapması için kullanacağız
# eğitilmiş modeli kullanarak test veri seti üzerinde tahmin yapıyoruz. Test veri setinden "Id" ve "SalePrice" sütunları çıkarıyoruz çünkü bu tahminleri yapmak gereksizdir.
# "Id" sütunu, her veri noktasının benzersiz kimliğidir ve "SalePrice" hedef değişkendir, tahmin edilmeye çalışılan değerdir.
predictions = model.predict(test_df.drop(["Id","SalePrice"], axis=1))

dictionary = {"Id":test_df.index, "SalePrice":predictions}   # bir sözlük oluşturduk. Bu sözlük, test veri setinin indekslerini "Id" olarak ve tahmin edilen değerleri "SalePrice" olarak içeriyor. Bu, her tahminin hangi evle ilişkili olduğunu belirlemek için kullanılır.
dfSubmission = pd.DataFrame(dictionary)  #  sözlüğü bir pandas DataFrame'ine dönüştürüyoruz
dfSubmission.to_csv("housePricePredictions.csv", index=False)  #  tahmin sonuçlarını "housePricePredictions.csv" adlı bir CSV dosyasına kaydediyoruz. index=False parametresi, DataFrame'in indekslerinin dosyaya kaydedilmemesini sağlar.





