############################################### İş Problemi##############################################
# ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak tahmin yapınız.


############################################### Veri Seti Hikayesi###############################################
# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır.
# İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını barındırmaktadır.
# 27.278 filmde 2.000.0263 derecelendirme içermektedir.
# Bu veri seti ise 17 Ekim 2016 tarihinde oluşturulmuştur.
# Bu veriler 138.493 kullanıcı tarafından 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasındaki veriler iiçermektedir.
# Kullanıcılar rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.

###############################################Değişkenler###############################################
# movie.csv:
# movieId – Eşsiz film numarası. (UniqueID)
# title – Film adı
# genres - Tür

# rating.csv:
# userid – Eşsiz kullanıcı numarası. (UniqueID)
# movieId – Eşsiz film numarası. (UniqueID)
# rating – Kullanıcı tarafından filme verilen puan
# timestamp – Değerlendirme tarihi

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)

#############################################
# Adım 1: Verinin Hazırlanması
#############################################

movie = pd.read_csv(r'4 -) RECOMMENDATION PROBLEMS/dataset/movie.csv')
rating = pd.read_csv(r'4 -) RECOMMENDATION PROBLEMS/dataset/rating.csv')

movie.head()
rating.head()

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape # 20.000.797


# Adım 3: Toplam oy kullanılma sayısı 1000'un altında olan filmleri veri setinden çıkarınız.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
comment_counts =  pd.DataFrame(df["title"].value_counts())

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
rare_movies = comment_counts[comment_counts["count"] <= 1000].index
# Ve veri setinden çıkartıyoruz:
common_movies = df[~df["title"].isin(rare_movies)]

common_movies.shape # 17.766.015



# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarak da ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()


# Adım 5 : Yapılan tüm işlemleri fonksiyonlaştırın:
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv(r'C:\Users\NerminB\PycharmProjects\pythonProject1\dataset\movie.csv')
    rating = pd.read_csv(r'C:\Users\NerminB\PycharmProjects\pythonProject1\dataset\rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

user_movie_df.head()

#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################
# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
random_user = 108170

# sample metodu ile:
# random_user = user_movie_df.sample().index[0]


# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()


# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
movies_watched

len(movies_watched)


#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################
# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
user_movie_df.head(3)

movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()

movies_watched_df.shape[1]


# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
user_movie_count = movies_watched_df.T.notnull().sum() # satırda eksik olmayan değerlerin sayısını verir
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# seçili kullanıcımıza en yakın kullanıcıya bakalım:
user_movie_count.sort_values(by="movie_count", ascending=False)


# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturunuz.
benzer = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > benzer]["userId"]
len(users_same_movies) # 2326 kişi benzer kullanıcı


#############################################
# Görev4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################
# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head(5)
final_df.shape


# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()


# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users = top_users.sort_values(by='corr', ascending=False) # en yüksek korelasyondan başlayarak sırala
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.head()
top_users.shape


# Adım 4:  top_users dataframe’ine rating veri setini merge ediniz
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings.head()
top_users_ratings["userId"].nunique()


#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################
# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()


# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()


# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False).head()



# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movies_to_be_recommend.merge(movie[["movieId", "title"]])



#############################################
# Adım 6: Item-Based Recommendation
#############################################
# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170


# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv(r'C:\Users\NerminB\PycharmProjects\pythonProject1\dataset\movie.csv')
rating = pd.read_csv(r'C:\Users\NerminB\PycharmProjects\pythonProject1\dataset\rating.csv')

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]


# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.

# movie_id ye sahip olan fiilm hangisi?
movie[movie["movieId"] == movie_id]["title"].values[0]

user_movie_df.head()

movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df.head()

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)



# Son iki adımı uygulayan fonksiyon
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)


# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’i öneri olarak veriniz.
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index # 0 ıncı index kendisi
