#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


data = pd.read_csv('country.csv')
data


# ##  Country.csv dosyasının özelliği
# Bu tablo, çeşitli ülkelerle ilgili bir dizi demografik, ekonomik ve coğrafi veriyi içermektedir. Tabloda her bir satır bir ülkeyi temsil ederken, sütunlar bu ülkelerle ilgili farklı özellikleri gösterir. İşte sütunların anlamları:
# 
# Country: Ülkenin adı.  
# Region: Ülkenin bulunduğu bölge (örneğin, Asya, Doğu Avrupa).  
# Population: Ülkenin toplam nüfusu.  
# Area (sq. mi.): Ülkenin yüzölçümü (mil kare olarak).  
# Pop. Density (per sq. mi.): Nüfus yoğunluğu (mil kare başına düşen kişi sayısı).  
# Coastline (coast/area ratio): Sahil uzunluğunun, ülkenin toplam alanına oranı.  
# Net migration: Net göç oranı (göçmenlerin ülkeye giren veya ülkeden çıkan kişi sayısına göre oranı).  
# Infant mortality (per 1000 births): Bebek ölüm oranı (1000 doğum başına).  
# GDP ($ per capita): Kişi başına düşen Gayri Safi Yurtiçi Hasıla (GSYİH).  
# Literacy (%): Okur-yazarlık oranı.  
# Phones (per 1000): Her 1000 kişi başına düşen telefon sayısı.  
# Arable (%): Tarıma elverişli arazi yüzdesi.  
# Crops (%): Ekilebilir ürünlerin yüzdesi.  
# Other (%): Diğer arazi kullanımı yüzdesi.  
# Climate: Ülkenin iklim kategorisi (numerik bir değer olarak gösterilmiş).  
# Birthrate: Doğum oranı.  
# Deathrate: Ölüm oranı.  
# Agriculture: Tarım sektörünün ekonomideki payı.  
# Industry: Sanayi sektörünün ekonomideki payı.  
# Service: Hizmet sektörünün ekonomideki payı.  
# 

# ## Bu Dosyada Yapacağınız görevleri alt taraftan bakabilirsiniz.

# ## 1. Görev : Nüfusa Göre Azalan Sırada Sıralama:

# In[4]:
import matplotlib.pyplot as plt
import seaborn as sns

# Nüfusa Göre Azalan Sırada Sıralama kodunu buraya yazınız


pop_sorted = data.sort_values(by="Population", ascending=False).head(10) ##sort_values ile azalan sırayla sıralama yaptım. Bu satırdaki amaç en kalabalık 10 ülkeyi bulmak.
print("\nNüfusa göre ilk 10 ülke:\n", pop_sorted[["Country","Population"]])
plt.figure(figsize=(8,4))
sns.barplot(x="Population", y="Country", data=pop_sorted, palette="magma" )
plt.title("Nüfusa Göre İlk 10 Ülke")
plt.show()


# ## 2. Görev: GDP per capita sütununa göre ülkeleri artan sırada sıralamak(Kişi başına düşen Gayri Safi Yurtiçi Hasıla).

# In[5]:
    
gdp_sorted = data.sort_values(by="GDP ($ per capita)", ascending=True).head(10) ##ascending=True ile artan sıralama yaptım. Bu satırdaki amaç kişi başına gelir en düşük 10 ülkeyi bulmak.
print("\nGDP per capita en düşük 10 ülke:\n", gdp_sorted[["Country","GDP ($ per capita)"]])
    
plt.figure(figsize=(8,4))
sns.barplot(x="GDP ($ per capita)", y="Country", data=gdp_sorted, palette="viridis" )
plt.title("GDP per capita En Düşük 10 Ülke")
plt.show()
# GDP per capita sütununa göre ülkeleri artan sırada sıralamak(Kişi başına düşen Gayri Safi Yurtiçi Hasıla). kodunu buradan yazınız.

# ## 3. Görev: Population sütunu 10 milyonun üzerinde olan ülkeleri seçmek.

# In[6]:
over_10m = data[data["Population"] > 10_000_000]
print("\nNüfusu 10 Milyondan Fazla Ülkeler", over_10m[["Country","Population"]].head(10))
over_10m_top10 = over_10m.sort_values(by="Population", ascending=False).head(10)
plt.figure(figsize=(8,4))
sns.barplot(x="Population", y="Country", data=over_10m_top10, palette="crest")
plt.title("Nüfusu 10 Milyondan fazla İlk 10 Ülke")
plt.show()


# ## 4. Görev: Literacy (%) sütununa göre ülkeleri sıralayıp, en yüksek okur-yazarlık oranına sahip ilk 5 ülkeyi seçmek.

# In[7]: 
top5_literacy = data.sort_values(by="Literacy (%)", ascending=False).head(5)
print("\nEn Yüksek Okur-Yazarlık oranına Sahip 5 Ülke", top5_literacy[["Country","Literacy (%)",]])
plt.figure(figsize=(8,4))
sns.barplot(x="Literacy (%)", y="Country", data=top5_literacy, palette="Blues_r")
plt.title("En Yüksek Okuryazarlık oranına Sahip 5 Ülke")
plt.show


# ## 5. Görev:  Kişi Başı GSYİH 10.000'in Üzerinde Olan Ülkeleri Filtreleme: GDP ( per capita) sütunu 10.000'in üzerinde olan ülkeleri seçmek.

# In[8]:
gdp_over_10000= data[data["GDP ($ per capita)"] > 10000]
gdp_over_10000_top10 = gdp_over_10000.sort_values(by="GDP ($ per capita)", ascending=False).head(10)
print(gdp_over_10000[["Country", "GDP ($ per capita)"]])
plt.figure(figsize=(8,4))
sns.barplot(x="GDP ($ per capita)", y="Country", data=gdp_over_10000_top10, palette="pastel")
plt.title("Kişi Başı GSYİH 10000 $ üzerinde olan ülkeler (ilk 10)")
plt.show()


# ## Görev 6 : En Yüksek Nüfus Yoğunluğuna Sahip İlk 10 Ülkeyi Seçme:
# Pop. Density (per sq. mi.) sütununa göre ülkeleri sıralayıp, en yüksek nüfus yoğunluğuna sahip ilk 10 ülkeyi seçmek.

# In[ ]:
top10_density = data.sort_values(by="Pop. Density (per sq. mi.)", ascending=False).head(10)
print("\nEn Yüksek Nüfus Yoğunluğuna Sahip İlk 10 Ülke:\n", top10_density[["Country", "Pop. Density (per sq. mi.)"]])
plt.figure(figsize=(8,4))
sns.barplot(x="Pop. Density (per sq. mi.)", y="Country", data=top10_density,palette="coolwarm")
plt.title("En Yüksek Nüfus Yoğunluğuna Sahip 10 Ülke")


plt.show





