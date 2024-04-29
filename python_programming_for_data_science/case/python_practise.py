###############################################
# Python Alıştırmalar
###############################################

###############################################
# GÖREV 1: Veri yapılarının tipleriniz inceleyiniz.
###############################################

x = 8
type(x)

y = 3.2
type(y)

z = 8j + 18
type(z)

a = "Hello World"
type(a)

b = True
type(b)

c = 23 < 22
type(c)


l = [1, 2, 3, 4, "String", 3.2, False]
type(l)
# Sıralıdır
# Kapsayıcıdır
# Değiştirilebilir


d = {"Name": "Jake",
     "Age": [27, 56],
     "Adress": "Downtown"}
type(d)
# Değiştirilebilir
# Kapsayıcı
# Sırasız
# Key değerleri farklı olacak


t = ("Machine Learning", "Data Science")
type(t)
# Değiştirilemez
# Kapsayıcı
# Sıralı


s = {"Python", "Machine Learning", "Data Science", "Python"}
type(s)
# Değiştirilebilir
# Sırasız + Eşsiz
# Kapsayıcı



###############################################
# GÖREV 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
###############################################

text = "The goal is to turn data into information, and information into insight."
text.upper().replace(",", " ").replace(".", " ").split()

# Mehmet Büyükgümüş
text_split = text.split()  # Mevcut metni parçalara ayırıyorum
new_text = []  # İstenilen çıktıyı kaydedeceğim boş bir liste oluşturuyorum
for word in text_split:  # Parçalara ayırdığım metnin her bir elemanı üzerinde döngü yaratıyorum
    for letter in word:  # Harflerin üzerinde dönecek yeni bir döngü daha yaratıyorum.
        if letter == ",":
            letter.replace(",", " ")
        elif letter == ".":
            letter.replace(".", " ")  # Nokta ve virgül karakterlerini boş ile değiştiriyorum
    new_text.append(word.upper())  # Her bir kelimeyi büyüterek yeni listeye kaydediyorum
print(new_text)  # Finalde istenen çıktıyı yazdırıyorum

# Sevinç Kabak
print(text.upper().replace(",", " ").replace(".", " ").upper().split())

# Yiğit Buğra Urun
a = list(text.replace(',',"").replace('.',"").upper().split())

# Said Yılmaz
text.upper().replace(",", " ").replace(".", " ").split()

# Serdinç
result = [word.upper() for word in text.replace(",.", " ").split()]
print(result)


###############################################
# GÖREV 3: Verilen liste için aşağıdaki görevleri yapınız.
###############################################

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakın.
len(lst)

# Adım 2: Sıfırıncı ve onuncu index'teki elemanları çağırın.
lst[0]
lst[10]

# Adım 3: Verilen liste üzerinden ["D","A","T","A"] listesi oluşturun.

data_list = lst[0:4]
data_list

# Adım 4: Sekizinci index'teki elemanı silin.

lst.pop(8)
lst

# Adım 5: Yeni bir eleman ekleyin.

lst.append(101)
lst


# Adım 6: Sekizinci index'e  "N" elemanını tekrar ekleyin.

lst.insert(8, "N")
lst


###############################################
# GÖREV 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
###############################################

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}


# Adım 1: Key değerlerine erişiniz.

dict.keys()

# Adım 2: Value'lara erişiniz.

dict.values()

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict.update({"Daisy": ["England", 13]})
dict

dict["Daisy"][1] = 14
dict


# Adım 4: Key değeri Ahmet value değeri [Turkey, 24] olan yeni bir
# değer ekleyiniz.

dict.update({"Ahmet": ["Turkey", 24]})
dict

# Adım 5: Antonio'yu dictionary'den siliniz.

dict.pop("Antonio")
dict



###############################################
# GÖREV 5: Arguman olarak bir liste alan,
# listenin içerisindeki tek ve çift sayıları ayrı listelere atayan
# ve bu listeleri return eden fonskiyon yazınız.
###############################################

l = [2, 13, 18, 93, 22]


def func(list:lst):
    cift_list = []
    tek_list = []

    for i in list:
        if i % 2 == 0:
            cift_list.append(i)
        else:
            tek_list.append(i)
    return cift_list, tek_list

cift, tek = func(l)

# Ahmet Can
list = [1,2,3,4,5,6,7,8]


tekList = []
ciftList = []

[ciftList.append(num) if num % 2 == 0 else tekList.append(num) for num in list ]

tekList
ciftList

# Esra Dalmızrak
list = [2, 13, 18, 93, 22]
A = []
B = []
for number in list:
    if number % 2 == 0:
        A.append(number)
    else:
        B.append(number)

print(A, B)

# Yasin
l = [2,13,18,93,22]

#♻️SOLUTION🪄

def func(l):
    #Listenin içerisindeki çift sayılardan oluşan liste.
    even_list = [num for num in l if num %2 == 0]
    # Listenin içerisindeki tek sayılardan oluşan liste.
    odd_list = [num for num in l if num %2 != 0 ]
    return even_list, odd_list

even_list, odd_list = func(l)



# İlayda Balık
l = [2, 13,18, 93, 22]

odd_list = []
even_list = []

def sep(list):
        for i in list:
                if i % 2 == 0:
                        even_list.append(i)
                else:
                        odd_list.append(i)

        return odd_list, even_list

sep(l)

# Alican Datlı
def even_or_odd(l):
    return [number for number in l if number % 2 == 0], [number for number in l if number % 2 == 1]

# Sevinç Kabak
def func(list):
    even_list = []
    odd_list = []
    for i in range(len(list)):
        if list[i] % 2 == 0:
            even_list.append(list[i])
        else:
            odd_list.append(list[i])
    return even_list, odd_list

# Yiğit Buğra Urun
def func(list): odd_list=[] even_list=[] for i in list: if i % 2 == 0: even_list.append(i) else: odd_list.append(i) return odd_list,even_list

# Mehmet Büyükgümüş
def tek_citt_sayi(sayilar):  # Fonksiyonu tanımlıyorum
    tek = []  # Tek sayılar için boş bir liste oluşturuyorum
    cift = []  # Çift sayılar için boş bir liste oluşturuyorum
    for number in sayilar:  # Tek ve çift sayıları tespit edebilmek için bir döngü yazıyorum.
        if number % 2 == 0:  # Şayet sayının sıfıra bölümübden kalan sıfır ise çift sayılar listesine ekliyorum
            cift.append(number)
        else:  # Şayet yukarıdaki koşul sağlanmadıysa tek sayılar listesine ekliyorum
            tek.append(number)
    return tek, cift  # Fonksiyonun çıktısı olan tek ve çift sayıları return ediyorum


tek_sayilar, cift_sayilar = tek_citt_sayi(sayilar)  # Yazdığım fonksiyonu çağırıyorum

# Said Yılmaz
def func(list):
    odd = []
    even = []

    for i in list:
        if i % 2 == 0:
            even.append(i)
        else:
            odd.append(i)
    return even, odd


# Serdinç
def func(sayilar):
    even_list = [sayi for sayi in sayilar if sayi % 2 == 0]
    odd_list = [sayi for sayi in sayilar if sayi % 2 != 0]
    return even_list, odd_list


even_list, odd_list = func(sayilar)

# Gökhan Arık
def func( l ):
    even_list = []
    odd_list = []
    for num in l:
        print(num)
        if num % 2 == 0:
            even_list.append(num)
        else:
            odd_list.append(num)
    return even_list, odd_list

even_list, odd_list = func(l)

# Efsun Türkyılmaz
l= [2,13,18,93,22]
even= []
odd= []
def func (list):
     for i in list:
         if i %2 == 0:
            even.append(i)
         else:
             odd.append(i)
     return even, odd


func(l)


###############################################
# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren
# öğrencilerin isimleri bulunmaktadır. Sırasıyla ilk üç öğrenci mühendislik fakültesinin
# başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
###############################################

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

for i, x in enumerate(ogrenciler):
    if i<3:
        i += 1
        print("Mühendislik Fakültesi", i,". öğrenci: ", x)
    else:
        i -= 2
        print("Tıp Fakültesi", i,". öğrenci: ", x)

###############################################
# GÖREV 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile
# bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır.
# Zip kullanarak ders bilgilerini bastırınız.
###############################################

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

for ders, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {kredi} olan {ders} kodlu dersin kontenjanı {kontenjan} kişidir.")

isim = 'oğuzhan'
yas = 25
print('Merhaba ' + isim + ' ' + str(yas) + ' yaşındasın.')
print(f'Merhaba {isim} {str(yas)} yaşındasın.')
y = 8.887238732
z = 3.2
print(f'Sonucunuz: {z:.2f}')
print(f'Sonucunuz: {y:.2f}')
print(f'deneme "" deneme ')
###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer
# kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız
# beklenmektedir.
###############################################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])


def kume(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))



# Eğer set1, set2'yi kapsıyorsa, set1 ve set2 kümelerinin kesişimini
# (yani her iki kümede de bulunan elemanları) ekrana yazdırır.
# Bu, set1.intersection(set2) kodu ile gerçekleştirilir.
###
# Eğer set1, set2'yi kapsamıyorsa, set2 kümesinin set1 kümesinden farkını
# (yani set2'de bulunan ancak set1'de bulunmayan elemanları) ekrana yazdırır.
# Bu, set2.difference(set1) kodu ile gerçekleştirilir.


kume(kume1, kume2)



