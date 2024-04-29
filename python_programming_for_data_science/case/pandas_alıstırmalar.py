##################################################
# Pandas Alıştırmalar
##################################################


import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df = sns.load_dataset("titanic")
df.head()
df.shape

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()


#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################


df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df["pclass"].unique()


#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################


df[["pclass","parch"]].nunique()


#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################


df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
df.info()


#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df["embarked"].value_counts()
df[df["embarked"] == "C"]

df[df["embarked"]=="C"].head(10)


#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################


df[df["embarked"] != "S"].head(10)



df[df["embarked"] != "S"]["embarked"].unique()



df[~(df["embarked"] == "S")]["embarked"].head(10)



#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df["age"]<30)   & (df["sex"]=="female")].head()

# Ali Ceylan
df.loc[(df["age"] < 30) & (df["sex"] == "female")]

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df["fare"] > 500 ) | (df["age"] > 70 )].head()

# aycan
df.loc[(df["fare"] > 500) | (df["age"] > 70)].head()


#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().sum()


#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df = df.drop("who", axis=1)
df.drop("who", axis=1, inplace= True)
df.head()



#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

df["deck"].value_counts()
df["deck"].mode()

type(df["deck"].mode())
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df.isnull().sum()


#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################
import matplotlib.pyplot as plt
df["age"].hist()
plt.show()

plt.hist(df["age"])

df["age"].fillna(df["age"].median(),inplace=True)
df.isnull().sum()



#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})
df.pivot_table(values="survived",index=["pclass","sex"], aggfunc=["sum","count","mean"])

#yasin
survival_stats = df.groupby(["pclass", "sex"])["survived"].agg(["sum", "count", "mean"])
survival_stats


#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

def age_30(age):
    if age < 30:
        return 1
    else
        return 0

df["age_flag"] = df["age"].apply(lambda x: 1 if x<30 else 0)
df.head()


# yasin
def age_flag_function(age):
    return 1 \
        if age < 30 \
        else 0

age_flag_function(28)

df["age_flag"]= df["age"].apply(lambda x: age_flag_function(x))
df["age_flag"]




#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################


df = sns.load_dataset("tips")
df.head()
df.shape


#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
df["time"].value_counts()

df.groupby("time").agg({"total_bill": ["sum","min","mean","max"]})


#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby(["day","time"]).agg({"total_bill": ["sum","min","mean","max"]})

# yasin
dF3 = tips.groupby(["day", "time"])["total_bill"].agg(["sum", "min", "max", "mean"])


#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################



df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum","min","max","mean"],
                                                                           "tip":  ["sum","min","max","mean"]})


#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################
df.loc[(df["size"] < 3) & (df["total_bill"] >10 ) , "total_bill"].mean()

# yasin
dl = tips[(tips["time"] == "Lunch") & (tips["sex"] == "Female")].groupby("day")[["total_bill","tip"]].agg(["sum", "min", "max", "mean"])

fo = df.loc[(df["size"] < 3) & (df["total_bill"] > 10)]
avgfo = fo["total_bill"].mean()
avgfo

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()



#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################


new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape


# farklı yol
new_df = df.sort_values("total_bill_tip_sum", ascending=False)
new_df[:30]


# sevinç
# df_new.sort_values(by="total_bill_tip_sum", ignore_index=True, ascending=False, inplace=True)
#
# df_new_first_thirty = df_new.iloc[0:31, :]

# yasin
# sorted_tips = tips.sort_values (by="total_bill_tip_sum", ascending=False)
# t30 = sorted_tips.head(30).copy()
# t30.head(5)


#  çiğdem
df.sort_values(by="total_bill_sum", ascending=False)
new_df = (df.sort_values(by="total_bill_sum", ascending=False)).head(30)