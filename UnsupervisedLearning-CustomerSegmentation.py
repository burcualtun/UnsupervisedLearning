

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
#Adım1
df= pd.read_csv("W7_8_9/W9_HW/data_20k.csv")
df.head()

#master_id silebiliriz.

#Adım2:  Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
#Not: Tenure (Müşterininyaşı), Recency (enson kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz

#Tarihi düzelt
df.loc[:,df.columns.str.contains("date")]=df.loc[:,df.columns.str.contains("date")].apply(lambda x: pd.to_datetime(x))


today_date = dt.datetime(2021, 6, 20)
df.head()
df.drop(['master_id'],axis=1,inplace=True)
#Tenure:todate-firstorderdate
df["NEW_TENURE"]=(today_date-df["first_order_date"]).dt.days.astype('int16')
#Recency:todate-lastorderdate
df["NEW_RECENCY"]=(today_date-df["last_order_date"]).dt.days.astype('int16')
#Total Order Count
df['NEW_TOTAL_ORDER_COUNT']=df['order_num_total_ever_online']+df['order_num_total_ever_offline']
#TOTAL VALUE
df['NEW_TOTAL_VALUE']=df['customer_value_total_ever_offline']+df['customer_value_total_ever_online']
#AVG TOTAL VALUE
df['NEW_AVG_TOTAL_VALUE']=df['NEW_TOTAL_VALUE']/df['NEW_TOTAL_ORDER_COUNT']
#AVG ONLINE TOTAL VALUE
df['NEW_AVG_ONLINE_TOTAL_VALUE']=df['customer_value_total_ever_online']/df['order_num_total_ever_online']
#AVG OFFLINE TOTAL VALUE
df['NEW_AVG_OFFLINE_TOTAL_VALUE']=df['customer_value_total_ever_offline']/df['order_num_total_ever_offline']
# ORDER count per Day
df['NEW_TOTAL_ORDER_COUNT_PER_DAY']=df['NEW_TOTAL_ORDER_COUNT']/(df['last_order_date']-df['first_order_date']).dt.days.astype('int16')
#Order value per day
df['NEW_TOTAL_VALUE_PER_DAY']=df['NEW_TOTAL_VALUE']/(df['last_order_date']-df['first_order_date']).dt.days.astype('int16')
#Recency Segmentation
df["NEW_RECENCT_CAT"] = pd.qcut(df['NEW_RECENCY'], 5, labels=[5, 4, 3, 2, 1])
#Frquency Segmentation
df["NEW_TOTAL_ORDER_COUNT_CAT"] = pd.qcut(df['NEW_TOTAL_ORDER_COUNT'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
#Monetary Segmentation
df["NEW_TOTAL_VALUE_CAT"] = pd.qcut(df['NEW_TOTAL_VALUE'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
#RFM Segment
df["NEW_RFM_SCORE"]=pd.to_numeric(df['NEW_RECENCT_CAT'].astype(str) +df['NEW_TOTAL_ORDER_COUNT_CAT'].astype(str))
df.head()
df.info()

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.loc[df['NEW_TOTAL_VALUE_PER_DAY']==np.inf,'NEW_TOTAL_VALUE_PER_DAY']=0
df.loc[df['NEW_TOTAL_ORDER_COUNT_PER_DAY']==np.inf,'NEW_TOTAL_ORDER_COUNT_PER_DAY']=0

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(df, cat_cols, drop_first=True)
dff.head()

dff.drop(['first_order_date','last_order_date','last_order_date_online','last_order_date_offline','interested_in_categories_12'],inplace=True,axis=1)
dff.head()



###########################################Görev2#########################################
#Adım1 K_MEANS - değişkenleri standartlaştırınız

sc = MinMaxScaler((0, 1))
df_std = sc.fit_transform(dff)


kmeans = KMeans(n_clusters=4, random_state=17).fit(df_std)
kmeans.get_params()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df_std)
elbow.show()
#opt değer
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df_std)

#Cluster'ları scale edilmemiş data içerisine ekliyoruz
clusters_kmeans = kmeans.labels_
dff["cluster"] = clusters_kmeans
dff["cluster"] = dff["cluster"] + 1
dff.head()


##Hiyerarşik kümeleme

hc_average = linkage(df_std, "average")

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

################################
# Kume Sayısını Belirlemek
################################


plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=7, linkage="average")
clusters = cluster.fit_predict(df_std)

dff["hi_cluster_no"] = clusters
dff["hi_cluster_no"] = dff["hi_cluster_no"] + 1

dff.head()

dff[dff['cluster']==5].head()





