#####################
### Beer Recommender
#####################


#Using user/beer ratings data from BeerAdvocate.com, constructs a model to predict user ratings
#for new beers.  Uses approximate matrix-factorization method trained with stochastic
#gradient descent to calibrate algorithm. 


import timeit
import os
import sys
import re
import math
import pandas
import numpy as np
import random


##### Define functions #####


# imports first N lines
def printlines(file, n):
    with open(file) as text2:
        head = [next(text2) for x in xrange(n)]
    for x in range(len(head)):
        print "Line " + str(x+1)
        print head[x]

# gets file length
def file_len(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i+1

# reads in file and makes list of words
def getwords(file, n):
    with open(file) as text2:
        head = [next(text2) for x in xrange(n)]
        for i, item in enumerate(head):
            head[i] = re.sub('\n', '', item)
    return head

# methods for importing and processing
class process_data:
    def __init__(self, filename, seednum, sampsize):
        self.filename = filename
        self.seednum = seednum
        self.sampsize = sampsize
    def readin(self):
        f = self.filename
        fl = file_len(f)
        raw = pandas.read_table(f, sep=",", nrows = fl)
        raw = raw[['review_overall', 'review_profilename', 'beer_name']]
        raw = raw.drop_duplicates(['review_profilename','beer_name'])
        return raw
    def getsample(self, raw):
        random.seed(self.seednum)
        rows = random.sample(raw.index, self.sampsize)
        raw2 = raw.ix[rows]
        return raw2
    def train_test(self, raw2):
        np.random.seed(self.seednum)
        raw2['is_train'] = np.random.uniform(0, 1, len(raw2)) < 0.9
        train, test = raw2[raw2['is_train']==True], raw2[raw2['is_train']==False]
        train, test = train.drop('is_train', 1), test.drop('is_train', 1)         
        return train, test

# train with SGD
def sgd(n_iter, df, U, M, alpha, beta, mu, lamb, lamb_a, lamb_b, eta, eta_a, eta_b):
    for _ in range(n_iter):
        print _
        for count, rating in enumerate(df['review_overall']):
            Vij = rating
            i = df.ix[count,3]  
            j = df.ix[count,4]
            # extract Ui, Mj
            Ui = U[i,]
            Mj = M[:,j]    
            p_ui_mj = np.dot(Ui,Mj)
            p_ui_mj_a_b = mu + alpha[i] + beta[j] + p_ui_mj    # added bias terms
            # calculate gradients
            grad_ui = (Vij - p_ui_mj_a_b)*Mj -lamb*Ui          # vector
            grad_mj = (Vij - p_ui_mj_a_b)*Ui -lamb*Mj          # vector
            grad_ai = (Vij - p_ui_mj_a_b) - lamb_a*alpha[i]    # scalar
            grad_bj = (Vij - p_ui_mj_a_b) - lamb_b*beta[j]     # scalar        
            # update parameters
            Ui = Ui + eta*grad_ui
            Mj = Mj + eta*grad_mj
            alpha[i] = alpha[i] + eta_a*grad_ai
            beta[j] = beta[j] + eta_b*grad_bj
            # update row/column in matrix
            U[i,] = Ui   
            M[:,j] = Mj    
    return U, M
            

##### Load and Process Data #####


os.chdir("C:\\Downloads\\Work\\SI 671\\Project\\beer_reviews_data")
name = "beer_reviews.txt"
xyz = process_data(name, 100, 100000)
data1 = xyz.readin()                    #1561732
data2 = xyz.getsample(data1)
train, test = xyz.train_test(data2)     #89939, 10061 

# remove users/beers in the test set that are not in training set        
check = train['review_profilename'].unique()
for i in test['review_profilename'].unique():
    if i not in check:
        print "not found", i
        test = test[test.review_profilename != i]
    else:
        print "found"
trainbeers =  numpy.sort(train['beer_name'].unique())
testbeers =  numpy.sort(test['beer_name'].unique())   
for i in testbeers:
    if i not in trainbeers:
        print "not found"
        test = test[test.beer_name != i]
    else:                                               # 8607         
        print "found"

# preprocess training set
df = train
unique_users = df['review_profilename'].unique()
ids = np.arange(len(unique_users))
cols = ['review_profilename', 'user_id']
data = {'review_profilename': unique_users, 'user_id': ids}
users = pandas.DataFrame(data, columns=cols)
unique_beers = df['beer_name'].unique()
ids = np.arange(len(unique_beers))
cols = ['beer_name', 'beer_id']
data = {'beer_name': unique_beers, 'beer_id': ids}
beers = pandas.DataFrame(data, columns=cols)

df = df.sort(['review_profilename', 'beer_name'], ascending=[True, False])
df = df.merge(users, how = "inner", on=["review_profilename"])
df = df.sort(['beer_name', 'review_profilename'], ascending=[True, False])
df = df.merge(beers, how = "inner", on=["beer_name"])
df = df.reindex(np.random.permutation(df.index))        # randomize observed ratings 
df = df.reset_index(drop=True)


##### Set Parameters #####


np.random.seed(100)
# choose number of features
k = 90
# learning rate
eta = 0.01
# regularization penalty
lamb = 0.02
# number of iterations for convergence
n_iter = 500
# initialize starting values for U, M: draw from U(0, 0.2)
num_users = len(unique_users)
num_items = len(unique_beers)
U = 0.2*np.random.random((num_users, k))         
M = 0.2*np.random.random((k, num_items))
# NEW: global average
mu = df['review_overall'].sum() /  float(len(df))
# NEW: regularization penalties for biases
lamb_a = 0.02
lamb_b = 0.02
# NEW: learning rate of biases
eta_a = 0.001
eta_b = 0.001
# NEW: initialize values for user and item bias (fixed or dynamic?)
alpha = np.random.uniform(-0.1, 0.1, num_users)     # users
beta = np.random.uniform(-0.1, 0.1, num_items)      # items


##### Stochastic Gradient Descent #####


# run model
start = timeit.default_timer() 
U, M = sgd(n_iter, df, U, M, alpha, beta, mu, lamb, lamb_a, lamb_b, eta, eta_a, eta_b)
stop = timeit.default_timer()
print stop - start     

# spot checks    
df.ix[100:105,]
i = 8811
j = 12587
Ui = U[i,]
Mj = M[:,j]
mu + alpha[i] + beta[j] + np.dot(Ui,Mj)


##### Evaluation #####
    

# training RMSE
Vhat = np.dot(U,M)
sse = 0
for count, rating in enumerate(df['review_overall']): 
    Vij = rating
    i = df.ix[count,3]
    j = df.ix[count,4]
    squared_error = math.pow(((mu + alpha[i] + beta[j] + Vhat[i,j]) - Vij), 2)
    sse += squared_error
    
mse = sse / float(len(df['review_overall']))
rmse = math.sqrt(mse)

# test RMSE
train_user_id = df[['review_profilename', 'user_id']]
train_user_id = train_user_id.drop_duplicates(['review_profilename','user_id'])
train_beer_id = df[['beer_name', 'beer_id']]
train_beer_id = train_beer_id.drop_duplicates(['beer_name','beer_id'])

test = test.merge(train_user_id, how = "inner", on=["review_profilename"])
test = test.merge(train_beer_id, how = "inner", on=["beer_name"])

sse = 0
for count, rating in enumerate(test['review_overall']): 
    Vij = rating
    i = test.ix[count,3]
    j = test.ix[count,4]
    squared_error = math.pow(((mu + alpha[i] + beta[j] + Vhat[i,j]) - Vij), 2)
    sse += squared_error
    
mse_test = sse / float(len(test['review_overall']))
rmse_test = math.sqrt(mse_test)

# baseline model 1
item_means = df.groupby(['beer_name']).agg(['mean'])
item_means.reset_index(level=0, inplace=True)
item_avgs = item_means[('review_overall', 'mean')].tolist()
item_beers = item_means[('beer_name', '')].tolist()
data = {'beer_name': item_beers, 'mean_beer_rating': item_avgs}
mean_beer_ratings = pandas.DataFrame(data, columns=['beer_name', 'mean_beer_rating'])

user_counts = df.groupby(['review_profilename']).agg(['count'])
user_counts.reset_index(level=0, inplace=True)
user_c = user_counts[('review_overall', 'count')].tolist()
user_profiles = user_counts[('review_profilename', '')].tolist()
data = {'review_profilename': user_profiles, 'user_num_ratings': user_c}
user_ratings_counts = pandas.DataFrame(data, columns=['review_profilename', 'user_num_ratings'])

df = df.merge(mean_beer_ratings, how = "inner", on = "beer_name")
df = df.merge(user_ratings_counts, how = "inner", on = "review_profilename")

df['bias'] = df['review_overall'] - df['mean_beer_rating']
calcs = df.groupby('review_profilename')
calcs2 = calcs.mean()
calcs2.reset_index(level=0, inplace=True)
avg_user_bias = calcs2[['review_profilename', 'bias']]

test = test.merge(avg_user_bias, how = "inner", on = "review_profilename")
test = test.merge(mean_beer_ratings, how = "inner", on = "beer_name")
test['baseline_pred'] = test['bias'] + test['mean_beer_rating']
test['squared_error'] = np.square((test['review_overall'] - test['baseline_pred']))
sse = test['squared_error'].sum()

mse_test_baseline = sse / float(len(test['review_overall']))
rmse_test_baseline = math.sqrt(mse_test_baseline)

# baseline model 2
df['user_count'] = 1
grouped = df.groupby('review_profilename')
sumUsers = grouped.sum()
sumUsers.reset_index(level=0, inplace=True)
sumUsers.rename(columns={'review_overall':'users_overall'}, inplace=True) 

df['beer_count'] = 1 
grouped = df.groupby('beer_name')
sumBeers = grouped.sum()
sumBeers.reset_index(level=0, inplace=True)
sumBeers.rename(columns={'review_overall':'beers_overall'}, inplace=True) 
sumBeers = sumBeers.drop('user_count', 1)

test = test.merge(sumUsers, how = "inner", on = "review_profilename")
test = test.merge(sumBeers, how = "inner", on = "beer_name")

test['avg_user'] = test['users_overall'] / test['user_count']  
test['avg_beer'] = test['beers_overall'] / test['beer_count']

# choose formula
test['predicted'] = (test['avg_user'] + test['avg_beer']) / 2
#test['predicted'] = (test['users_overall'] + test['beers_overall']) / (test['user_count'] + test['beer_count'])
#test['predicted'] = test['avg_user']
test['predicted'] = test['avg_beer']

test['squared_error'] = np.square((test['review_overall'] - test['predicted']))
sse = test['squared_error'].sum()
mse_test_baseline = sse / float(len(test['review_overall']))
rmse_test_baseline = math.sqrt(mse_test_baseline)
