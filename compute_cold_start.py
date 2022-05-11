import os

from operator import itemgetter
from numpy import random, zeros
import numpy as np
from scipy.linalg import norm
from math import log, exp
import matplotlib.pyplot as plt

def compute_user_rank(user_item_ratings):

    user_rank = {}
    for user_id in user_item_ratings.keys():
        for item_id in user_item_ratings[user_id]:
            user_rank[user_id] = user_rank.get(user_id, 0.0) + 1.0

    user_rank_list = []
    for user_id in user_rank.keys():
        user_rank_list.append((user_id, user_rank[user_id]))

    user_rank_list = sorted(user_rank_list, key=itemgetter(1))

    user_rank_dict = {}
    for user_pair in user_rank_list:
        user_rank_dict[user_pair[0]] = user_pair[1]

    return user_rank_dict

def compute_item_rank(user_item_ratings):

    item_rank = {}
    for user_id in user_item_ratings.keys():
        for item_id in user_item_ratings[user_id]:
            item_rank[item_id] = item_rank.get(item_id, 0.0) + 1.0

    item_rank_list = []
    for item_id in item_rank.keys():
        item_rank_list.append((item_id, item_rank[item_id]))

    item_rank_list = sorted(item_rank_list, key=itemgetter(1))

    item_rank_dict = {}
    for item_pair in item_rank_list:
        item_rank_dict[item_pair[0]] = item_pair[1]

    return item_rank_dict

def compute_cold_start_0(train_data_dict, test_data_dict, item_rank_dict, eta):

    user_len = max(train_data_dict.keys()) + 1
    item_len = max(item_rank_dict.keys()) + 1

    print(user_len)
    print(item_len)

    u = random.random([user_len, 30])
    v = random.random([item_len, 30])

    user_id_list = random.random_sample(100) * train_data_dict.keys().__len__()
    user_repo = list(train_data_dict.keys())

    user_features = {}
    item_features = {}

    for iter_no in range(0, 30):
    
        for uid_id in user_id_list:

            user_id = user_repo[int(uid_id)]
            
            if train_data_dict[user_id].__len__() < 1:
                continue

            item_id_list = list(train_data_dict[user_id].keys())
            sampled_item_id_list = random.random_sample(10) * item_id_list.__len__()

            item_repo = train_data_dict[user_id].keys()

            for iid in sampled_item_id_list:

                item_id = item_id_list[int(iid)]

                R_v = train_data_dict[user_id][item_id_list[int(iid)]]
    
                u[user_id] += eta*v[item_id]/np.dot(u[user_id], v[item_id]) - 2*eta*u[user_id]
                v[item_id] += eta*u[user_id]/np.dot(u[user_id], v[item_id]) - 2*eta*v[item_id]

                user_features[user_id] = u[user_id]
                item_features[item_id] = v[item_id]

    pr_dict = {}
    pr_list = []

    mae = 0.0
    total_no = 0.0

    R_max = -1.0
    for user_id in test_data_dict.keys():
        for item_id in test_data_dict[user_id]:
            if user_id in user_features and item_id in item_features:
                R_v = np.dot(u[user_id], v[item_id])
                if R_v > R_max :
                    R_max = R_v
 
    FILE = open('test_data_dict.txt', 'w')
    for user_id in test_data_dict.keys():
        for item_id in test_data_dict[user_id]:
            if user_id in user_features and item_id in item_features:
                R_v = 5.0 * (np.dot(u[user_id], v[item_id])/R_max)
                #R_v = 5.0 * random.random()
                pr_dict[item_id] = pr_dict.get(item_id, 0)+1
                FILE.write('%s\n' % R_v)
                mae += abs(R_v - test_data_dict[user_id][item_id])
                total_no += 1
    FILE.close()

    for item_id in pr_dict.keys():
        pr_list.append((item_id, pr_dict[item_id]))

    pr_list_s = sorted(pr_list, key=itemgetter(1), reverse=True)
    rank_list = []

    iter_id = 0
    rank_id = 1
    while iter_id < pr_list_s.__len__():
        rank_list.append(rank_id)
        while iter_id+1 < pr_list_s.__len__() and pr_list_s[iter_id] == pr_list_s[iter_id+1]:
            rank_list.append(rank_id)
            iter_id += 1
        rank_id += 1
        iter_id += 1


    DME = 0.0
    for rank_val in rank_list:
        DME += log(rank_val*1.0/rank_list[-1])
    DME = 1 + rank_list.__len__()/DME

    return (mae/total_no, DME)

def compute_cold_start_1(test_data_dict, eta):

    pr_dict = {}

    mae = 0.0
    total_no = 0.0

    FILE = open('test_data_dict.txt', 'w')
    for user_id in test_data_dict.keys():
        for item_id in test_data_dict[user_id]:
            R_v = 5.0 * random.random()
            pr_dict[item_id] = pr_dict.get(item_id, 0)+1
            FILE.write('%s\n' % R_v)
            mae += abs(R_v - test_data_dict[user_id][item_id])
            total_no += 1
    FILE.close()

    pr_list = []
    for item_id in pr_dict.keys():
        pr_list.append((item_id, pr_dict[item_id]))

    pr_list_s = sorted(pr_list, key=itemgetter(1), reverse=True)
    rank_list = []

    iter_id = 0
    rank_id = 1
    while iter_id < pr_list_s.__len__():
        rank_list.append(rank_id)
        while iter_id+1 < pr_list_s.__len__() and pr_list_s[iter_id] == pr_list_s[iter_id+1]:
            rank_list.append(rank_id)
            iter_id += 1
        rank_id += 1
        iter_id += 1

    DME = 0.0
    for rank_val in rank_list:
        DME += log(rank_val*1.0/rank_list[-1])
    DME = 1 + rank_list.__len__()/DME

    print('DME:%s' % DME)

    return (mae/total_no, DME)

def compute_mf(train_data_dict, test_data_dict, item_rank_dict, eta):

    user_len = max(train_data_dict.keys()) + 1
    item_len = max(item_rank_dict.keys()) + 1

    u = random.random([user_len, 30])
    v = random.random([item_len, 30])

    user_id_list = random.random_sample(100) * train_data_dict.keys().__len__()
    user_repo = list(train_data_dict.keys())

    user_features = {}
    item_features = {}

    for iter in range(0, 30):
    
        for uid_id in user_id_list:

            user_id = user_repo[int(uid_id)]
            
            if train_data_dict[user_id].__len__() < 1:
                continue

            item_id_list = list(train_data_dict[user_id].keys())
            sampled_item_id_list = random.random_sample(10) * item_id_list.__len__()

            item_repo = train_data_dict[user_id].keys()

            for iid in sampled_item_id_list:

                item_id = item_id_list[int(iid)]

                R_v = train_data_dict[user_id][item_id_list[int(iid)]]
    
                u[user_id] += eta*2*(R_v - np.dot(u[user_id], v[item_id])) * v[item_id]
                v[item_id] += eta*2*(R_v - np.dot(u[user_id], v[item_id])) * u[user_id]

                user_features[user_id] = u[user_id]
                item_features[item_id] = v[item_id]

    pr_dict = {}
    pr_list = []

    mae = 0.0
    total_no = 0.0

    FILE = open('test_data_dict.txt', 'w')
    for user_id in test_data_dict.keys():
        for item_id in test_data_dict[user_id]:
            if user_id in user_features and item_id in item_features:
                R_v = 5.0 * (np.dot(u[user_id], v[item_id])/(norm(u[user_id])*norm(v[item_id])))
                pr_dict[item_id] = pr_dict.get(item_id, 0)+1
                FILE.write('%s\n' % R_v)
                mae += abs(R_v - test_data_dict[user_id][item_id])
                total_no += 1
    FILE.close()

    for item_id in pr_dict.keys():
        pr_list.append((item_id, pr_dict[item_id]))

    pr_list_s = sorted(pr_list, key=itemgetter(1), reverse=True)
    rank_list = []

    iter_id = 0
    rank_id = 1
    while iter_id < pr_list_s.__len__():
        rank_list.append(rank_id)
        while iter_id+1 < pr_list_s.__len__() and pr_list_s[iter_id] == pr_list_s[iter_id+1]:
            rank_list.append(rank_id)
            iter_id += 1
        rank_id += 1
        iter_id += 1

    DME = 0.0
    for rank_val in rank_list:
        #print('%s, %s\n' %(rank_val, rank_list[-1]))
        DME += log(rank_val*1.0/rank_list[-1])
    DME = 1 + rank_list.__len__()/DME

    print('DME:%s' % DME)

    return (mae/total_no, DME)

def predict_mf(test_data_dict, u, v):

    pr_dict = {}
    pr_list = []

    mae = 0.0
    total_no = 0.0

    FILE = open('test_data_dict.txt', 'w')
    for user_id in test_data_dict.keys():
        for item_id in test_data_dict[user_id]:
            if user_id in u and item_id in v:
                R_v = 5.0 * (np.dot(u[user_id], v[item_id])/(norm(u[user_id])*norm(v[item_id])))
                pr_dict[item_id] = pr_dict.get(item_id, 0)+1
                FILE.write('%s\n' % R_v)
                mae += abs(R_v - test_data_dict[user_id][item_id])
                total_no += 1
    FILE.close()

    for item_id in pr_dict.keys():
        pr_list.append((item_id, pr_dict[item_id]))

    pr_list_s = sorted(pr_list, key=itemgetter(1), reverse=True)
    rank_list = []

    iter_id = 0
    rank_id = 1
    while iter_id < pr_list_s.__len__():
        rank_list.append(rank_id)
        while iter_id+1 < pr_list_s.__len__() and pr_list_s[iter_id] == pr_list_s[iter_id+1]:
            rank_list.append(rank_id)
            iter_id += 1
        rank_id += 1
        iter_id += 1

    DMF = 0.0
    for rank_val in rank_list:
        DMF += log(rank_val*1.0/rank_list[-1])
    DMF = 1 + rank_list.__len__()/DMF

    print('ZMF DMF:%s' % DMF)
    mae /= total_no
    return (mae, DMF)

if __name__ == '__main__':

    input_file = 'ml-1m/ratings_new.dat'
    #input_file = 'ml-latest-small/ratings.csv'

    user_item_ratings = {}

    user_set = set()
    item_set = set()

    with open(input_file, 'r') as FILE:
        for line in FILE:
            user_pair = line.split(',')
            user_id = user_pair[0]
            item_id = user_pair[1]
            user_set.add(user_id)
            item_set.add(item_id)

    user_list = list(user_set)
    item_list = list(item_set)

    user_dict = {}
    item_dict = {}

    for idx, user_id in enumerate(user_list):
        user_dict[user_id] = idx

    for idx, item_id in enumerate(item_list):
        item_dict[item_id] = idx

    with open(input_file, 'r') as FILE:
        for line in FILE:
            user_pair = line.split(',')
            user_id = user_pair[0]
            item_id = user_pair[1]
            rating = float(user_pair[2])
            user_item_ratings.setdefault(user_dict[user_id], {})
            user_item_ratings[user_dict[user_id]][item_dict[item_id]] = rating

    train_set = {}
    test_data_list = []

    train_set_list = []
    test_set_list = []

    train_set_dict = {}
    test_set_dict = {}

    for user_id in user_item_ratings.keys():
        item_list = [item_id for item_id in user_item_ratings[user_id].keys()]
        train_set.setdefault(user_id, [])
        train_set_dict.setdefault(user_id, {})
        test_set_dict.setdefault(user_id, {})
        if item_list.__len__() > 4:
            train_set[user_id] = item_list[:-2]
            for item_id in item_list[:-2]:
                train_set_list.append((user_id, item_id, user_item_ratings[user_id][item_id]))
                train_set_dict[user_id][item_id] = user_item_ratings[user_id][item_id]
            for x in range(-2, 0):
                test_data_list.append((user_id, item_list[x]))
                test_set_list.append((user_id, item_list[x], user_item_ratings[user_id][item_list[x]]))
                test_set_dict[user_id][item_list[x]] = user_item_ratings[user_id][item_list[x]]

    with open('train_set.txt', 'w') as FILE:
        for data_rec in train_set_list:
            FILE.write('%s\t%s\t%s\n'%(data_rec[0], data_rec[1], data_rec[2]))

    with open('test_set.txt', 'w') as FILE:
        for data_rec in test_set_list:
            FILE.write('%s\t%s\t%s\n'%(data_rec[0], data_rec[1], data_rec[2]))


    user_rank_dict = compute_user_rank(user_item_ratings)
    item_rank_dict = compute_item_rank(user_item_ratings)

    eta_list = [5e-4, 6e-4, 0.001, 0.002, 0.003, 0.004,0.005]
    mae_0_list = []
    mae_1_list = []
    mae_list = []
    ZMF_0_list = []
    ZMF_1_list = []
    MF_list = []
    for eta in eta_list: 
        (mae_0, ZMF_0) = compute_cold_start_0(train_set_dict, test_set_dict, item_rank_dict, eta) 
        (mae_1, ZMF_1) = compute_cold_start_1(test_set_dict, eta)
        (mae, MF) = compute_mf(train_set_dict, test_set_dict, item_rank_dict, eta)
        print('MAE comparison (ZeroMat, Random Placement, Classic MF): %s %s %s' % (mae_0, mae_1, mae)) 
        mae_0_list.append(mae_0)
        mae_1_list.append(mae_1)
        mae_list.append(mae)
        ZMF_0_list.append(ZMF_0)
        ZMF_1_list.append(ZMF_1)
        MF_list.append(MF)
        print('Degree of Matthew Effect comparison (ZeroMat, Random Placement, Classic MF): %s %s %s' % (ZMF_0, ZMF_1, MF))

    plt_0, = plt.plot(eta_list, mae_0_list)
    plt_1, = plt.plot(eta_list, mae_1_list)
    plt_2, = plt.plot(eta_list, mae_list)
    plt.legend([plt_0, plt_1, plt_2], ['ZeroMat', 'Random Placement', 'Classic Matrix Factorization'])
    plt.xlabel('Gradient Learning Step')
    plt.ylabel('MAE')
    plt.show()

   
    plt_0, = plt.plot(eta_list, ZMF_0_list)
    plt_1, = plt.plot(eta_list, ZMF_1_list)
    plt_2, = plt.plot(eta_list, MF_list)
    plt.legend([plt_0, plt_1, plt_2], ['ZeroMat', 'Random Placement', 'Classic Matrix Factorization'])
    plt.xlabel('Gradient Learning Step')
    plt.ylabel('Degree of Matthew Effect')
    plt.show()

