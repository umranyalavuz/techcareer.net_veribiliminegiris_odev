#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('dava_sonuclari.csv')
data.head()


# ## VERİ SETİ İNCELEME : 
# Case Type: Davanın türü (Criminal, Civil, Commercial)  
# Case Duration (Days): Davanın süresi (gün olarak)  
# Judge Experience (Years): Hakimin deneyim yılı  
# Number of Witnesses: Tanık sayısı  
# Legal Fees (USD): Hukuk masrafları (USD olarak)  
# Plaintiff's Reputation: Davacının itibarı (1: Düşük, 2: Orta, 3: Yüksek)  
# Defendant's Wealth (USD): Davalının serveti  
# Number of Evidence Items: Delil sayısı  
# Number of Legal Precedents: İlgili hukuki emsal sayısı  
# Settlement Offered (USD): Teklif edilen uzlaşma miktarı  
# Severity: Davanın ciddiyet derecesi (1: Düşük, 2: Orta, 3: Yüksek)  
# Outcome: Davanın sonucu (0: Kaybetmek, 1: Kazanmak)  

# ## Görevler
# 
# ### Veri Ön İşleme:
# * Veri setini inceleyin ve eksik veya aykırı değerler olup olmadığını kontrol edin.  
# * Gerektiğinde eksik verileri doldurun veya çıkarın.  
# * Özelliklerin ölçeklendirilmesi gibi gerekli veri dönüşümlerini uygulayın. 
# 
# ### Veri Setini Ayırma:
# * Veri setini eğitim ve test setleri olarak ayırın (örn. %80 eğitim, %20 test).  
# 
# ### Model Kurulumu:
# * Karar ağacı modelini oluşturun ve eğitim verileri üzerinde eğitin.
# 
# ### Modeli Değerlendirme:
# * Test verilerini kullanarak modelin doğruluğunu değerlendirin.
# * Doğruluk, precision, recall ve F1-score gibi performans metriklerini hesaplayın.
# 
# ### Sonuçları Görselleştirme:
# * Karar ağacının yapısını görselleştirin.
# * Karar ağacının nasıl çalıştığını ve hangi özelliklerin davanın sonucunu belirlemede en etkili olduğunu açıklayın.

# In[1]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split #veriyi eğitim ve test olarak ayırdım.
from sklearn.tree import DecisionTreeClassifier, plot_tree #decisiontreeclassfier ile karar ağacı modeli kurdum.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix #accuracy ile doğruluk oranını, precision ile kesinlik oranını, recall ile duyarlılık oranını verdim. f1 score ile precision ile recall un dengeli ortalamasını verdim, classifiction report ile bu matrixlerin hepsini tablo şeklinde gösterdim. son olarak confusion matrix ile 2x2 tablo ile doğru yanlış tahminlerini analiz ettim.
# In[2]:
# CSV dosyasını okudum ve ilk 5 satırı ekrana verdim.
data = pd.read_csv("dava_sonuclari.csv")
print("İlk 5 satır:")
print(data.head())


# In[3]:
# Eksik değerleri kontrol ettim
print("\nEksik değer kontrolü:")
print(data.isnull().sum)

# Gerekirse eksikleri doldurma/çıkarma (burada basit doldurma örneği) yaptım.
data=data.dropna()

# In[4]:
# Kategorik değişkenleri encode et (Case Type gibi)
data=pd.get_dummies(data, drop_first=True) #drop_first=True: gereksiz sütun bırakmamak için.

# Bağımsız değişkenler ve bağımlı değişkeni ayır
X= data.drop("Outcome", axis=1) #Bağımsız değişkenler(X)= sonucu etkileyen faktörler
y= data["Outcome"] #Bağımlı değişken (y) tahmin etmek istediğimiz sonuç.

# In[5]:
# Eğitim ve test ayırma (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
#veriyi %80 eğitim, %20 test olarak ayırıyoruz.
#random_state=42 aynı sonucu tekrar elde etmek için kullandım
#stratify=y eğiitm ve test setinde kazanan/kaybeden oranını dengede tutmak için kullandım.
# In[6]:
# Karar ağacı modeli
clf= DecisionTreeClassifier(random_state=42, max_depth=5) #ağacın en fazla 5 seviye dallanması için kullandım.
clf.fit(X_train, y_train) # fiti ise eğitim verileri ile modeli eğitmek için kullandım.

# Tahmin
y_pred= clf.predict(X_test) #test verilerini kullanarak tahmin yaptım.

# In[7]:
# Metrikler   burda kaldınnnnnn devam
acc = accuracy_score(y_test, y_pred) #doğru tahminlerin tüm tahminlere oranı
prec = precision_score(y_test, y_pred) #pozitif tahminlerin doğru olma oranı
rec = recall_score(y_test, y_pred) #gerçek pozitiflerin kaçına ualştık
f1 = f1_score(y_test, y_pred) #precision ve recall dengesi

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)

print("\nClassification Report:\n", classification_report(y_test, y_pred,)) #precision, recall, f1 ve destek değerlerini tablo halinde verdim. 

# Confusion Matrix çizdim.
cm = confusion_matrix(y_test, y_pred,)
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=["Kaybetti", "Kazandı"], yticklabels=["Kaybetti", "Kazandı"]) #sns.heatmap ile ısı haritası çizdim. annot=true ile kutuların içine sayıları yazdım. fmt="d" ile sayıları tam sayı yaptım. 
plt.title("Confusion Matrix")
plt.ylabel("Gerçek")
plt.xlabel("Tahmin")
plt.show()


# In[8]:
# Karar ağacını görselleştirme
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=["Kaybetti", "Kazandı"], filled=True, fontsize=2) #feature_names=X.columns ile ağacın dallarında hangi özellik kullanıldığını yazdım. class_names ile yapraklarda sınıf isimelrini yazdım. filled ile kutu renklerini sınıfa göre boyadım.
plt.show()

# In[9]:
# Özellik önemleri
importances = pd.DataFrame({
    "Özellik": X.columns,
    "Önem Skoru": clf.feature_importances_
}).sort_values(by="Önem Skoru", ascending=False)

print("\nÖzelliklerin Önemi:")
print(importances)

sns.barplot(data=importances, x="Önem Skoru", y="Özellik", palette="viridis")
plt.title("Özellik Önemleri")
plt.show()



