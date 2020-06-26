import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NL        # library for visualization
import random                              # pseudo-random number generator
from os import getcwd
import numpy as np
# downloads sample twitter dataset. uncomment the line below if running on a local machine.

class SentimentPredict():
    ## 去除不必要的字符
    def process_tweet(self,tweet):
        from nltk.stem import PorterStemmer
        from nltk.corpus import stopwords
        import re
        import string
        from nltk.tokenize import TweetTokenizer
        stemmer = PorterStemmer()#利用Porter stemming algorithm来进行stemming
        stopwords_english = stopwords.words('english')#加载英文的stop words词库
        ## 抄一波代码，原课程提供的utils.py中的
        # remove stock market tickers like $GE
        tweet = re.sub(r'\$\w*', '', tweet)
        # remove old style retweet text "RT"
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # remove hyperlinks
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        # remove hashtags
        # only removing the hash # sign from the word
        tweet = re.sub(r'#', '', tweet)
        # tokenize tweets
        tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                                reduce_len=True)
        tweet_tokens = tokenizer.tokenize(tweet)

        tweets_clean = []
        for word in tweet_tokens:
            if (word not in stopwords_english and  # remove stopwords
                    word not in string.punctuation):  # remove punctuation
                # tweets_clean.append(word)
                stem_word = stemmer.stem(word)  # stemming word
                tweets_clean.append(stem_word)

        return tweets_clean

    ## frequency dictionary
    def build_freqs(self,tweets,ys):
        yslist = np.squeeze(ys).tolist()
        freqs = {}
        for y,tweet in zip(yslist,tweets):
            for word in self.process_tweet(tweet):
                pair = (word,y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1
        return freqs

    # 顾名思义
    def sigmoid(self,z):
        h = 1/(1+np.exp(-z))#按照表达式
        return h
    
    ## 梯度下降法更新theta
    def gradientDescent(self,x,y,theta,alpha,num_iters):# alpha为学习率，越小每次改变的就越小
        m = len(x) ## 输入数据行数

        for i in range(0,num_iters):
            z = x.dot(theta)
            h = self.sigmoid(z)##计算h函数结果
            ## 计算损失函数
            J = (-1)/m * (y.transpose().dot(np.log(h))+(1-y).transpose().dot(np.log(1-h)))
            ## 更新 theta
            theta -= alpha/m*(x.transpose().dot(h-y))
        
        J = float(J)
        return J,theta

    ## 特征提取
    def extract_features(self,tweet,freqs):
        word_1 = self.process_tweet(tweet)
        # x = [1,sum pos, sum neg]
        x = np.zeros((1,3))
        x[0,0]=1

        for word in word_1:
            if(word,1) in freqs.keys():
                x[0,1] += freqs[(word,1)]#在pos里出现的频率
            if(word,0) in freqs.keys():
                x[0,2] += freqs[(word,0)]#在neg里出现的频率
        
        assert(x.shape == (1,3))# 判断是否异常
        return x

    ## 进行预测
    def predict_tweet(self,tweet):
        ## 抽取特征
        x = self.extract_features(tweet,self.freqs)

        ## 进行预测
        y_pred = self.sigmoid(x.dot(self.theta))

        return y_pred

    def test_logistic_regression(self,test_x,test_y):
        y_hat=[]
        for tweet in test_x:
            y_pred = self.predict_tweet(tweet)
            if y_pred>0.5:
                y_hat.append(1)
            else:
                y_hat.append(0)
        accuracy = sum(sum(np.array(y_hat) == np.array(test_y.transpose())))/len(y_hat)
        return accuracy
    
    ## 进行训练
    def train_logistic_regression(self,train_x,train_y):
        self.freqs = self.build_freqs(train_x,train_y)#建立起frequency dictionary
        X = np.zeros((len(train_x),3))
        for i in range(len(train_x)):
            X[i,:] = self.extract_features(train_x[i],self.freqs) # 第i个语料的向量x放入X的第i行
        Y = train_y

        self.J,self.theta = S_Pre.gradientDescent(X,Y,np.zeros((3,1)),1e-9,1500)## 进行1500次迭代


## 加载数据集
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# print("positive tweets数量:",len(all_positive_tweets))

# print("negative tweets数量:",len(all_negative_tweets))

# print(all_positive_tweets[0])#来随便看一个text内容大概是啥样子

# 80%做训练集，20%做测试集
test_pos = all_positive_tweets[4000:]#最后1000做测试集
test_neg = all_negative_tweets[4000:]
train_pos = all_positive_tweets[:4000]#前4000做训练集
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg # 8000,前4000为1，后4000为0
test_x = test_pos + test_neg # 2000，前1000为1，后1000为0

# 创建训练集对应的标签array
train_y = np.append(np.ones((len(train_pos),1)),np.zeros((len(train_neg),1)),axis = 0)

# 创建测试集对应的标签array
test_y = np.append(np.ones((len(test_pos),1)),np.zeros((len(test_neg),1)),axis = 0)



S_Pre = SentimentPredict()
## 训练模型
S_Pre.train_logistic_regression(train_x,train_y)


## 使用模型来判断text的情感
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, S_Pre.predict_tweet(tweet)))

tmp_accuracy = S_Pre.test_logistic_regression(test_x,test_y)
print(tmp_accuracy)


## 自定义tweet并进行判断

my_tweet = 'That"s fucking Great!'
y_hat = S_Pre.predict_tweet(my_tweet)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')