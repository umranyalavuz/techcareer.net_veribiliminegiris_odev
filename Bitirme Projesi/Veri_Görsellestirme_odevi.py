#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data = pd.read_csv('50_Startups.csv')


# In[10]:


data.head()


# ## Bu veri seti, 50 farklı startup şirketinin çeşitli harcamalarını ve kârlılıklarını içermektedir.  
# 
# R&D Spend (Ar-Ge Harcaması): Şirketin araştırma ve geliştirme (Ar-Ge) için harcadığı tutar.  
# Administration (Yönetim Harcaması): Şirketin yönetim giderleri için harcadığı tutar.  
# Marketing Spend (Pazarlama Harcaması): Şirketin pazarlama ve reklam faaliyetleri için harcadığı tutar. 
# State (Eyalet): Şirketin faaliyet gösterdiği eyalet (örneğin, New York, California, Florida).  
# Profit (Kâr): Şirketin elde ettiği toplam kâr.  
# Bu veri seti, startup'ların çeşitli harcama kalemleri ile kârlılıkları arasındaki ilişkileri analiz   etmek için kullanılabilir. Örneğin, Ar-Ge veya pazarlama harcamalarının kârlılık üzerindeki etkisini   incelemek için uygun bir veri setidir.  

# ## 1.GÖREV : R&D Harcaması ve Kâr Arasındaki İlişki (Scatter Plot): Ar-Ge harcamaları ile kâr arasındaki ilişkiyi gösteren bir dağılım grafiği.

# In[11]:
plt.figure(figsize=(8,5))
plt.scatter(data['R&D Spend'], data['Profit'], color='blue', alpha=0.9)
plt.title("R&D Harcaması ile Kâr İlişkisi")
plt.xlabel("R&D Harcaması")
plt.ylabel("Kar")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
# Kodu buraya yazınız 


# ## 2.GÖREV: Yönetim Harcamaları ve Kâr Arasındaki İlişki (Scatter Plot): Yönetim harcamaları ile kâr arasındaki ilişkiyi gösteren bir dağılım grafiği.

# In[12]:
plt.figure(figsize=(8,5))
plt.scatter(data['Administration'], data['Profit'], color='red', alpha=0.9)
plt.title("Yönetim Harcamaları ve Kâr Arasındaki İlişki")
plt.xlabel("Yönetim Harcaması")
plt.ylabel("Kar")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()



# ## 3. GÖREV: Eyaletlere Göre Ortalama Kâr (Bar Chart): Farklı eyaletlerdeki startup'ların ortalama kârlarını karşılaştıran bir çubuk grafik.

# In[13]:
state_means = data.groupby('State')['Profit'].mean()
plt.figure(figsize=(8,5))
state_means.plot(kind='bar', color=['skyblue','pink','purple'])
plt.title("Eyaletlere Göre Ortalama Kâr")
plt.xlabel("Eyalet")
plt.ylabel("Ortalama Kar")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# Kodu buraya yazınız 

# ## 4. GÖREV: Harcama Türlerinin Karşılaştırması (Boxplot): R&D, yönetim ve pazarlama harcamalarının dağılımını karşılaştıran bir kutu grafiği.

# In[14]:

plt.figure(figsize=(8,5))
plt.boxplot(
    [data['R&D Spend'], data['Administration'], data['Marketing Spend']],
    labels=['R&D Spend', 'Administration', 'Marketing Spend'])
plt.title("Harcama Türlerinin Karşılaştırması")
plt.xlabel("Yönetim Harcaması")
plt.ylabel("Harcama Miktarı")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()









# In[ ]:




