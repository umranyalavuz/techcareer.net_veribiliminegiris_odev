#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd

# In[5]:


data = pd.read_csv('dava.csv')
data


# ## Veri Seti inceleme
# Veri Seti Özellikleri:  
# Case Duration (Gün): Davanın tamamlanması için geçen süre (gün cinsinden).  
# Number of Witnesses (Tanık Sayısı): Dava boyunca dinlenen tanık sayısı.  
# Legal Fees (Hukuk Maliyetleri): Dava süresince oluşan toplam hukuk maliyetleri (USD cinsinden).  
# Number of Evidence Items (Delil Sayısı): Davada kullanılan delil sayısı.  
# Severity (Ciddiyet Düzeyi): Davanın ciddiyet düzeyi (1: Düşük, 2: Orta, 3: Yüksek).  
# Outcome (Sonuç): Davanın sonucu (0: Aleyhte, 1: Lehinde).  

# ## GÖREV: 
# Özellik Seçimi: Hangi özelliklerin kümeleme için kullanılacağına karar verin.  
# Küme Sayısını Belirleme: Elbow yöntemi gibi tekniklerle optimal küme sayısını belirleyin.  
# Kümeleme İşlemi: K-Means algoritmasını kullanarak verileri kümeleyin.  
# Sonuçları Görselleştirme: Kümeleme sonuçlarını uygun grafiklerle görselleştirin ve yorumlayın.  

# In[ ]:
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler

features = data[["Case Duration (Days)", "Number of Witnesses", "Legal Fees (USD)", "Number of Evidence Items" ]]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

inertia = []
K = range(1, 11)
for k in K :
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.plot(K, inertia, "bo-")
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("Intertia")
plt.title("Elboe Yöntemi")
plt.show()    

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(scaled_features)


# 6. Görselleştirme
sns.scatterplot(x="Case Duration (Days)", y="Legal Fees (USD)", 
                hue="Cluster", data=data, palette="viridis")
plt.title("K-Means Kümeleme (k=3)")
plt.show()
print("Küme Merkezleri (ölçeklenmiş):")
print(kmeans.cluster_centers_)



