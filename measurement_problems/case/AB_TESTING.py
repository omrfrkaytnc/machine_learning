# AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi ve average bidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# müşteri sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchase metriğine odaklanılmalıdır.


#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır.
# Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmıştır.

# DEĞİŞKENLER:
# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç

#####################################################
# Proje Görevleri
#####################################################

#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################
# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan
# veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)  # dataframe i tam genişlikte gösterir.
pd.set_option('display.float_format', lambda x: '%.5f' % x)

dataframe_control = pd.read_excel("3 -) MEASUREMENT PROBLEMS/datasets/ab_testing.xlsx", sheet_name="Control Group")
dataframe_test = pd.read_excel("3 -) MEASUREMENT PROBLEMS/datasets/ab_testing.xlsx", sheet_name="Test Group")

df_control = dataframe_control.copy()
df_test = dataframe_test.copy()


# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df_control)
check_df(df_test)

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak
# kontrol ve test grubu verilerini birleştiriniz.

df_control["group"] = "control"
df_test["group"] = "test"

df_control.head()
df_test.head()


df = pd.concat([df_control, df_test], axis=0, ignore_index=True)  # birleştirirken indexi kaldığı yerden devam etsin
df.head()
df.tail()
df.shape

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

# H0 : M1 = M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark yoktur.)
# H1 : M1!= M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında fark vardır.)


# Adım 2: Kontrol ve test grubu için purchase(satın alınan ürün sayısı) ortalamalarını analiz ediniz:

df.groupby("group").agg({"Purchase": "mean"})
# farklılık görülüyor ancak bu istatistiki bir anlam taşımakta mıdır?


#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.
# Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını
# Purchase değişkeni üzerinden ayrı ayrı test ediniz.

# Normallik Varsayımı :
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
# p-value < 0.05 H0 RED
# p-value > 0.05 H0 REDDEDİLEMEZ
# Test sonucuna göre normallik varsayımı kontrol ve test grupları için sağlanıyor mu?
# Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value = 0.5891
# HO reddedilemez. Control grubunun değerleri normal dağılım varsayımını sağlamaktadır.

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value = 0.1541
# HO reddedilemez. Test grubunun değerleri normal dağılım varsayımını sağlamaktadır.


# Varyans Homojenliği:
# Varyans; Verilerin ortalamadan olan sapmalarının karelerinin ortalamasıdır.
# Yani standart sapmanın karesidir.

# H0: Varyanslar homojendir.
# H1: Varyanslar homojen Değildir.
# p-value < 0.05 H0 RED
# p-value > 0.05 H0 REDDEDİLEMEZ
# Kontrol ve test grubu için varyans homojenliğinin sağlanıp sağlanmadığını
# Purchase değişkeni üzerinden test ediniz.
# Test sonucuna göre normallik varsayımı sağlanıyor mu?
# Elde edilen p-value değerlerini yorumlayınız.

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value = 0.1083
# HO reddedilemez. Control ve Test grubunun değerleri varyans homejenliği varsayımını sağlamaktadır.
# Varyanslar Homojendir.


# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.


# Varsayımlar sağlandığı için bağımsız iki örneklem t testi (parametrik test) yapılmalıdır.

# H0: M1 = M2 (Kontrol grubu ve test grubu satın alma ortalamaları arasında ist. ol.anl.fark yoktur.)
# p < 0.05 HO RED,
# p > 0.05 HO REDDEDİLEMEZ

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value = 0.3493


# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# p-value=0.3493
# H0 REDDEDİLEMEZ yani Kontrol grubu ve test grubu satın alma ortalamaları
# arasında ist. ol.anl.fark yoktur.
# iki grup ortalaması arasındki farklılıklar şans eseri ortaya çıkmıştır.


##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
# iki grupta da normallik varsayımı ve varyans homojenliği sağlandığı için "Bağımsız iki örneklem T testi" uygulanmıştır.
# p-value değerleri 0,05 den büyük olduğu gözlenmiş böylece H0 hipotezi reddedilememiştir.



# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
# purchase e göre yani "Tıklanan reklamlar sonrası satın alınan ürün sayısı" nda iki yöntem arasında istatistiki anlamda
# anlamlı bir fark olmadığından müşteri istediği yöntemi seçebilir. tıklanma, etkileşim, kazanç ve dönüşüm oranlarındaki
# farklılıklar değerlendirilip hangi yöntemin daha kazançlı olduğu tespit edilebilir.
# iki grup gözlenmeye devam edilebilir.
# Şimdilik iki yöntem arasındaki farklılık şans eseri ortaya çıkmıştır, diyebiliriz.






