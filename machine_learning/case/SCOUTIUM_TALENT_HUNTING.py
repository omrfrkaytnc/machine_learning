

"""**YAPAY ÖĞRENME İLE YETENEK AVCILIĞI SINIFLANDIRMA**

**İş Problemi**

Scoutlar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.

**Veriseti Hikayesi**

Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

**attributes:** Oyuncuları değerlendiren kullanıcıların bir maçta izleyip değerlendirdikleri her oyuncunun özelliklerine verdikleri puanları içeriyor. (bağımsız değişkenler)

**potential_labels:** Oyuncuları değerlendiren kullanıcıların her bir maçta oyuncularla ilgili nihai görüşlerini içeren potansiyel etiketlerini içeriyor. (hedef değişken)

**9 Değişken, 10730 Gözlem, 0.65 mb**
"""



"""Değişkenler

#####################################################

**scoutium_attributes.csv**

8 Değişken     10.730 Gözlem

**task_response_id:** Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi.

**match_id:** İlgili maçın id'si.

**evaluator_id:** Değerlendiricinin(scout'un) id'si.

**player_id:** İlgili oyuncunun id'si.

**position_id:** İlgili oyuncunun o maçta oynadığı pozisyonun id'si.

1- Kaleci

2- Stoper

3- Sağ bek

4- Sol bek

5- Defansif orta saha

6- Merkez orta saha

7- Sağ kanat

8- Sol kanat

9- Ofansif orta saha

10- Forvet

**analysis_id:** Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme.


**attribute_id:** Oyuncuların değerlendirildiği her bir özelliğin id'si.


**attribute_value:** Bir scoutun bir oyuncunun bir özelliğine verilen değer(puan).


**potential_label:** Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

########################################################

**scoutium_potential_labels.csv**

5 Değişken 322 Gözlem

**task_response_id   :** Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi

**match_id           :** İlgili maçın id'si

**evaluator_id       :** Değerlendiricinin(scout'un) id'si

**player_id          :** İlgili oyuncunun id'si

**potential_label    :** Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket.

(hedef değişken)
"""



import pandas as pd
import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
!pip install catboost
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# data frame in sınırlamalarını kaldırıyoruz

pd.set_option('display.max_columns', None)  # bütün sütunları göster
pd.set_option('display.max_rows', None)     # bütün satırları göster
pd.set_option('display.width', 500)  # sütunlar max 500 tane gösterilsin
pd.set_option('display.expand_frame_repr', False)  # çıktının tek bir satırda olmasını sağlar

"""GÖREVLER

Veri Setinin Hazırlanması

Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
"""

df_att = pd.read_csv("6 -)  MACHINE LEARNING/datasets/scoutium_attributes.csv", sep=";")
df_att.head()

df_pot = pd.read_csv("6 -)  MACHINE LEARNING/datasets/scoutium_potential_labels.csv", sep=";")
df_pot.head()

"""Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz. ("task_response_id", 'match_id', 'evaluator_id' "player_id"  4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)"""

df = pd.merge(df_att, df_pot, how='left', on=["task_response_id", 'match_id', 'evaluator_id', "player_id"])
df.head()

df.to_csv(r"scout_merged.csv", index=False, header=True)          # C:\Users\GunalHincal\Downloads\

"""Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.



"""

df.shape

# position_id sınıfı 1 dışında olan herşeyi df_scout a geri kaydet dedik, yani kaleci pozisyonlarını veri setimizden çıkarıyoruz
df = df[~(df["position_id"] == 1)]

# ya da bunu kullanabilirdik
df = df[df["position_id"] != 1]

df.shape  # (10030, 9) -- 700 satır gitmiş

"""Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.

( below_average sınıfı tüm verisetinin %1'ini oluşturur)
"""

df["potential_label"].value_counts()
# average          7922
# highlighted      1972
# below_average     136

df = df[~(df["potential_label"] == "below_average")]
df["potential_label"].value_counts()
# average        7922
# highlighted    1972

df["attribute_value"].value_counts()  #  Bir scoutun bir oyuncunun bir özelliğine verilen değer(puan).

df["attribute_id"].value_counts()  # Oyuncuların değerlendirildiği her bir özelliğin id'si.

df["attribute_id"].nunique()  # 34 adet özellik olduğunu görüyoruz

df["player_id"].value_counts()  # İlgili oyuncunun id'si

df["player_id"].nunique()  # 248 adet oyuncu görüyoruz

df.shape

# ya da ~ kullanmadan bu şekilde de yapabilirdik
df = df[df["potential_label"] != "below_average"]
df.shape  # (9894, 9)

"""Adım 5: Oluşturduğunuz verisetinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satır bir oyuncu olacak şekilde manipülasyon yapınız.

Adım 5-1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
"""

pivot = df.pivot_table(index=["player_id", "position_id", "potential_label"],
                             columns="attribute_id",
                             values="attribute_value")
pivot.head()

# ya da
pivot = pd.pivot_table(df, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])

"""Adım 5-2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini string e çeviriniz. (df.columns.map(str))"""

pivot.dtypes

pivot = pivot.reset_index(drop=False)

pivot.head()

pivot.dtypes  # indexleri de artık kolon olarak görmekteyiz

# sayısal olan kolon adlarını string e çeviriyoruz
pivot.columns = pivot.columns.map(str)
pivot.head()

pivot.to_csv(r"scout_pivot.csv", index=False, header=True)  # eğer yeni pivotlanmış dataframe i csv olarak kaydetmek isterseniz

"""Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz."""

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


labelEncoderCols = ["potential_label"]

for col in labelEncoderCols:
    pivot = label_encoder(pivot, col)

pivot.head()

pivot.tail()

pivot["potential_label"].head(20)
# alfabatik sıraya göre önce 0 ve sonra 1 gelecek şekilde encode yaptı
# average = 0, highlighted = 1

pivot.head()

pivot.dtypes

"""Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız."""

num_cols = pivot.columns[3:]

num_cols

# iki ifade de aynı (bu ifade liste olarak kaydeder sonucu, yukarıdaki ifade index olarak kaydeder,
# index olarak kaydeden kullanılırsa çıktıyı daha sonradan listeye çevirmek gerekir)
num_cols = [col for col in pivot.columns if col not in ["player_id", "position_id", "potential_label"]]

num_cols

"""# KEŞİFÇİ VERİ ANALİZİ"""

# Genel Resim
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(pivot)

"""Numerik ve kategorik değişkenleri inceleyiniz."""

# KATEGORİK DEĞİŞKENLERİN ANALİZİ
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in ["position_id", "potential_label"]:
    cat_summary(pivot, col)

pivot.head()

pivot["position_id"].value_counts()

# NUMERİK DEĞİŞKENLERİN ANALİZİ
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(pivot, col, plot=True)

"""Numerik değişkenler ile hedef değişken incelemesini yapınız."""

# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(pivot, "potential_label", col)

# KORELASYON ANALİZİ
pivot[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pivot[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# korelasyon iki değişken arasındaki ilişkinin yönünü ve derecesini gösterir



"""Feature Extraction uygulayın."""

pivot["min"] = pivot[num_cols].min(axis=1)
pivot["max"] = pivot[num_cols].max(axis=1)
pivot["sum"] = pivot[num_cols].sum(axis=1)
pivot["mean"] = pivot[num_cols].mean(axis=1)
pivot["median"] = pivot[num_cols].median(axis=1)

pivot["mentality"] = pivot["position_id"].apply(lambda x: "defender" if (x == 2) | (x == 5) | (x == 3) | (x == 4) else "attacker")

# 1- Kaleci
# 2- Stoper
# 3- Sağ bek
# 4- Sol bek
# 5- Defansif orta saha
# 6- Merkez orta saha
# 7- Sağ kanat
# 8- Sol kanat
# 9- Ofansif orta saha
# 10- Forvet

pivot.head()

pivot.tail()

"""Adım 6: Label Encoder fonksiyonunu kullanarak bu sefer de yeni oluşturduğumuz değişken olan "mentality” kategorisini (attacker, defender) sayısal olarak ifade ediniz."""

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


labelEncoderCols = ["mentality"]

for col in labelEncoderCols:
    pivot = label_encoder(pivot, col)

pivot.head()

"""Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız."""

num_cols = [col for col in pivot.columns if col not in ["player_id", "position_id", "potential_label"]]

num_cols

# ya da
# Adım 7 deki Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız. görevinde eğer
num_cols = pivot.columns[3:]   # bu fonksiyonu kullanarak num_cols nesnesi oluşturduysanız bu çıktı yukarıda bahsettiğimiz gibi index olarak geleceği için
# standartlaştırma öncesi listeye çevirme işlemini uygulayıp ondan sonra scale işlemi yapmamız gerekecek

num_cols = [col for col in pivot.columns if col not in ["player_id", "position_id", "potential_label"]]

# yeni oluşturduğumuz değişkenleri bir listeye atayıp bu listeyi de numerik kolonlar arasına ekliyoruz
pivot.head()
lst = ["min","max","sum","mean","median"]
num_cols = list(num_cols)

for i in lst:
    num_cols.append(i)


# yeni değişkenlerle beraber standart scaler ı uyguluyoruz.
scaler = StandardScaler()
pivot[num_cols] = scaler.fit_transform(pivot[num_cols])
pivot.head()
pivot[num_cols]
pivot.head()

"""# ADIM 8

Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
"""

ss = StandardScaler()

for i in num_cols:
    pivot[i] = ss.fit_transform(pivot[[i]])
num_cols = pivot.columns[3:]
pivot.head()

"""Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)"""

y = pivot["potential_label"]
X = pivot.drop(["potential_label", "player_id"], axis=1)

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   # ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]


    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


base_models(X, y, scoring="accuracy")

Base Models....
accuracy: 0.8633 (LR)
accuracy: 0.8155 (KNN)
accuracy: 0.8083 (CART)
accuracy: 0.8671 (RF)
accuracy: 0.845 (Adaboost)
accuracy: 0.8524 (GBM)
accuracy: 0.8597 (XGBoost)
accuracy: 0.8524 (LightGBM)

base_models(X, y, scoring="precision")

Base Models....
precision: 0.7354 (LR)
precision: 0.7778 (KNN)
precision: 0.5671 (CART)
precision: 0.7887 (RF)
precision: 0.6416 (Adaboost)
precision: 0.7143 (GBM)
precision: 0.7051 (XGBoost)
precision: 0.7211 (LightGBM)

base_models(X, y, scoring="recall")

Base Models....
recall: 0.5185 (LR)
recall: 0.1793 (KNN)
recall: 0.5721 (CART)
recall: 0.4639 (RF)
recall: 0.5867 (Adaboost)
recall: 0.5702 (GBM)
recall: 0.5692 (XGBoost)
recall: 0.5175 (LightGBM)

base_models(X, y, scoring="f1")

Base Models....
f1: 0.5888 (LR)
f1: 0.2813 (KNN)
f1: 0.578 (CART)
f1: 0.5858 (RF)
f1: 0.5966 (Adaboost)
f1: 0.5889 (GBM)
f1: 0.6204 (XGBoost)
f1: 0.5885 (LightGBM)

base_models(X, y, scoring="roc_auc")

Base Models....
roc_auc: 0.8416 (LR)
roc_auc: 0.7059 (KNN)
roc_auc: 0.7049 (CART)
roc_auc: 0.8875 (RF)
roc_auc: 0.7973 (Adaboost)
roc_auc: 0.8753 (GBM)
roc_auc: 0.8524 (XGBoost)
roc_auc: 0.8693 (LightGBM)

"""# EKSTRA AŞAMALAR

# Hyperparameter optimization
"""

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

# Bu fonksiyonda şunu yapıyoruz; öncesindeki hataya bak hiperparametre değerlerini bul,
# hiperparametre değerleri ile sonrasındaki hatasını bul şeklindedir. Çıktılarımız aşağıda buradan gözlemleyelim

def hyperparameter_optimization(X, y, cv=5, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


best_models = hyperparameter_optimization(X, y)

Hyperparameter Optimization....
########## KNN ##########
roc_auc (Before): 0.7469
roc_auc (After): 0.7469
KNN best params: {'n_neighbors': 5}

########## CART ##########
roc_auc (Before): 0.7159
roc_auc (After): 0.7226
CART best params: {'max_depth': 1, 'min_samples_split': 2}

########## RF ##########
roc_auc (Before): 0.8998
roc_auc (After): 0.9036
RF best params: {'max_depth': 15, 'max_features': 7, 'min_samples_split': 15, 'n_estimators': 200}

########## XGBoost ##########
roc_auc (Before): 0.8668
roc_auc (After): 0.8861
XGBoost best params: {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 100}

########## LightGBM ##########
roc_auc (Before): 0.8891
roc_auc (After): 0.8921
LightGBM best params: {'colsample_bytree': 1, 'learning_rate': 0.01, 'n_estimators': 500}

best_models = hyperparameter_optimization(X, y, cv=5, scoring="accuracy")

best_models = hyperparameter_optimization(X, y, cv=5, scoring="precision")

best_models = hyperparameter_optimization(X, y, cv=5, scoring="recall")

best_models = hyperparameter_optimization(X, y, cv=5, scoring="f1")

"""# Stacking & Ensemble Learning"""

# 3 modelin cross validation hatasına bakacağız 3 metrik açısından bunları değerlendirip ekrana print edeceğiz

# özetle şunu yapıyoruz, knn gel bakalım, rf gel bakalım, gbm, lightgbm, xgboost gel bakalım hepiniz bi model kurun
# hepiniz bi tahminde bulunun hepinizi bir araya getirdik bir gözlem birimi geldiğinde tahmini hep birlikte yapacağızdır.

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"Precision: {cv_results['test_precision'].mean()}")
    print(f"Recall: {cv_results['test_recall'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


voting_clf = voting_classifier(best_models, X, y)

Voting Classifier...
Accuracy: 0.8487179487179487
Precision: 0.8141025641025642
Recall: 0.3742690058479532
F1Score: 0.5013461538461539
ROC_AUC: 0.8753319407456216



"""# Prediction for a New Observation"""

import joblib

X.columns

# random user diyerek bir kullanıcı seçiyoruz
random_user = X.sample(1, random_state=45)

# voting_clf ile bu random user ı tahmin ediyoruz
voting_clf.predict(random_user)

# Çalışma kapandığında bu model uçmasın istiyorsak joblib.dumb ile modelimizi kaydediyoruz, isimlendirmesini yapıyoruz
joblib.dump(voting_clf, "scotium_voting.pkl")

# bunun çalışıp çalışmadığını nasıl anlarız, yine joblibi kullanıp yükle deriz bu sefer
new_model = joblib.load("scotium_voting.pkl")

# new_model olarak kaydetmiştik yüklediğimizi şimdi new model predict dersek random seçtiği bir kullanıcıyı tahmin eder
new_model.predict(random_user)



"""# Adım 10: Değişkenlerin önem düzeyini belirten feature_importance grafiğine bakınız"""

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


model = LGBMClassifier()
model.fit(X, y)

plot_importance(model, X)



