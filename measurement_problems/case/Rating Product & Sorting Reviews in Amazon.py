###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde
# hesaplanmasıdır. Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için
# ürünün öne çıkması ve satın alanlar için sorunsuz bir alışveriş deneyimi demektir.
# Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması olarak karşımıza çıkmaktadır.
# Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden hem maddi kayıp hem de müşteri
# kaybına neden olacaktır.
# Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler ise
# satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti, ürün kategorileri ile çeşitli meta dataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı


import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False) # dataframe i tam genişlikte gösterir.
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################

df = pd.read_csv('3 -) MEASUREMENT PROBLEMS/datasets/amazon_review.csv')
df.head()

# overall: Ürün rating’i
df["overall"].mean()
# ürünün ortalama puanı; 4.58


###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################
# overall: Ürün rating’i
# day_diff: Değerlendirmeden itibaren geçen gün sayısı

df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean()  # günümüze en yakın

df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].
                                                           quantile(0.50)), "overall"].mean()

df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].
                                                           quantile(0.75)), "overall"].mean()

df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean()  # günümüze en uzak, en eski


# zaman bazlı ortalama ağırlıkların belirlenmesi
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"]. \
        mean() * w1 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].
                       quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(
            0.50)), "overall"].mean() * w2 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].
                       quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(
            0.75)), "overall"].mean() * w3 / 100 + \
        dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].
                       quantile(0.75)), "overall"].mean() * w4 / 100


time_based_weighted_average(df, w1=28, w2=26, w3=24, w4=22)  # 4.59

df["overall"].mean()  # 4.58


# time_based_weighted_average(df, w1=50, w2=25, w3=15, w4=10)  # 4.63

# ******************************************************************************************
# alternatif:
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = df["reviewTime"].max()
df["recency_rating_review"] = (current_date - df["reviewTime"]).dt.days
df["recency_cut"]= pd.qcut(df["recency_rating_review"], 4, labels= ["q1", "q2", "q3", "q4" ])

def time_based_weighted_average(dataframe, w1=30, w2=28, w3=26, w4=16):
    return dataframe.loc[df["recency_cut"] == "q1", "overall"].mean() * w1 / 100 + \
           dataframe.loc[df["recency_cut"] == "q2", "overall"].mean() * w2 / 100 + \
           dataframe.loc[df["recency_cut"] == "q3", "overall"].mean() * w3 / 100 + \
           dataframe.loc[df["recency_cut"] == "q4", "overall"].mean() * w4 / 100
time_based_weighted_average(df)

###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up ve down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.


df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

df.head()


###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp değişkenler olarak
# Veriye Ekleyiniz
###################################################
# score_pos_neg_diff:
def score_up_down_diff(up, down):
    return up - down


df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)


# score_average_rating:
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)


# wilson_lower_bound:
# istatistiki olarak bir sıralama gerçekleştirelim:
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(5)

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)




# kalabalığın bilgeliği dikkate alınmış, total-vote yüksek

