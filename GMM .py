import pickle
import utils
import preprocess
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


def main():
    PathRoot = r"C:\Users\nycu_dev1\Desktop\YEN-HSU Code\formationtable\\"
    
    # 設定開始和結束日期
    start_date = datetime(2018, 10, 19)
    end_date = datetime(2018, 10, 19)#remember to change back

    # 用來儲存所有檔案的 DataFrame
    all_data = []

    # 從開始日期到結束日期逐天檢查檔案是否存在
    current_date = start_date
    while current_date <= end_date:
        file_path = PathRoot + current_date.strftime('%Y%m%d') + 'for150del16_AB.csv'
        
        
        if os.path.exists(file_path):
            pre_df = pd.read_csv(file_path)
            all_data.append(pre_df)  # 將讀取的 DataFrame 加入列表中
        current_date += timedelta(days=1)

    # 將所有 DataFrame 合併成一個
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        print("已合併所有存在的檔案")
    else:
        df = pd.DataFrame()
        print("沒有找到任何檔案")

    # 檢查結果
    print(df)
    
    #df = pd.read_csv(PathRoot+"20181019for150del16_AB.csv")
    filtered_df=df[df['Johansen_slope']==0]
    df=filtered_df[['Johansen_intercept', 'Johansen_std']]
    df=df.dropna()
    coordinates=np.column_stack((df['Johansen_intercept'], df['Johansen_std']))
    print(coordinates) 

    #train
    n_components = 3  #3 clusters
    gmm = GaussianMixture(n_components=n_components, random_state=42)

    row_count=coordinates.shape[0]
    tvt = int(0.8*row_count)
    print(row_count)
    print(tvt)

    gmm.fit(coordinates[:tvt])

    #test
    test_scores = gmm.score(coordinates[tvt:])  
    test_labels = gmm.predict(coordinates[tvt:])  
    test_silhouette = silhouette_score(coordinates[tvt:], test_labels)
    print(f"Log Likelyhood: {test_scores}")
    print(f"Silhoutte score: {test_silhouette}")

    # Plotting the stacked Johansen_intercept and Johansen_std pairs
    plt.figure(figsize=(10, 6))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='blue', label='Intercept vs Std', alpha=0.7)

    # Adding labels and title
    plt.xlabel('Johansen Intercept')
    plt.ylabel('Johansen Std')
    plt.title('Scatter Plot of Johansen Intercept vs Johansen Std')
    plt.ylim(0, 0.0025)
    plt.legend()
    plt.grid(True)
    plt.show()
    

if __name__=="__main__":
    main()


"""
with roughly 5000
(90, 10)
Log Likelyhood: 4.406855174296233
Silhoutte score: 0.6648467275306482

(80, 20)
Log Likelyhood: 4.393381228355523
Silhoutte score: 0.6641306608429643

(70, 30)
Log Likelyhood: 4.371505925211136
Silhoutte score: 0.6719501994526652

(60, 40)
Log Likelyhood: 4.438660649374624
Silhoutte score: 0.6780111550697034
"""