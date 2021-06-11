#ライブラリのインポート
from gensim.models import word2vec
from sklearn.decomposition import PCA #主成分分析器
import matplotlib.pyplot as plt

#モデルのパス
model_path = 'wiki.model'

#モデルの読み込み
model = word2vec.Word2Vec.load(model_path)

#入力
target_word = input('どの単語の類義語が見たいですか？:')
get_num = int(input('いくつ単語を表示したいですか？:'))

#類義語の分散表現をリストに格納
item = model.wv.most_similar(target_word,topn=get_num)
data = [] #類義語の分散表現を格納
words = [] #類義語名を格納
words.append([target_word,'r'])
data.append(model.wv[target_word])
for i in item:
    print(i)
    words.append([i[0],'b'])
    data.append(model.wv[i[0]])

#主成分分析により２次元に圧縮する
pca = PCA(n_components=2)
pca.fit(data)
data_pca= pca.transform(data)

#プロットの準備
fig=plt.figure(figsize=(10,6),facecolor='w')
plt.rcParams["font.size"] = 10

#プロット
i = 0
while i < len(data):
    #点プロット
    plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x", color=words[i][1])
 
    #文字プロット
    plt.annotate(words[i][0], (data_pca[i][0], data_pca[i][1]), size=12,color=words[i][1])
    i += 1

#グラフの出力
plt.show()