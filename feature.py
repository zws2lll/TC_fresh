#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
def fuc1():
    print '\n'

#
def fuc2():
    print '\n'

def fuc3():
    """用户在发生购买行为的前1（n）天将该商品加入购物车的概率。


    :return:
    """
    file1 = pd.read_csv(
        'data/fresh_comp_offline/tianchi_fresh_comp_train_user2.csv')
    file = file1.sort_index(by=['user_id', 'item_id', 'day_from_start',
                                    'operator'], ascending=False)
    file.reset_index(drop=True, inplace=True)
    file.to_csv('data/fresh_comp_offline/5.txt', index=False)
    buyCount = 0
    buyAfterPur = 0
    dict = {}
    print file
    for index, row in file.T.iteritems():
        if index == 0:
            continue
        if file.ix[index]['user_id'] != file.ix[index-1]['user_id']:
            dict[str(file.ix[index]['user_id'])] \
                = [str(file.ix[index]['user_id']), \
                       str(float(buyAfterPur) / buyCount)]
            buyAfterPur = 0
            buyCount = 0
        elif file.ix[index]['operator'] == 4:
            buyCount += 1
            if file.ix[index]['item_id'] == file.ix[index-1]['item_id'] \
                and file.ix[index-1]['day_from_start'] + 1 \
                    == file.ix[index]['day_from_start'] \
                and file.ix[index-1]['operator'] == 3:
                buyAfterPur += 1
    df = pd.DataFrame(data=dict)
    df = df.T
    print df
    print '\n'

def fuc4(start, end, trainData=True):
    """训练数据(已排序)
    [user_id, item_id, 近3天总操作次数, 近1天收藏、购物车次数, label(0或1)]
    :return:
    """
    START_FLAG = start
    END_FLAG = end
    LABLE_FLAG = end + 1
    file = pd.read_csv(
        'data/fresh_comp_offline/tianchi_fresh_comp_train_user2.csv')
    # file = pd.read_csv('data/fresh_comp_offline/test.csv')
    file.loc[len(file.index)] = np.array([0,0,END_FLAG,0])
    indexTmp = 0
    feature1 = []
    feature2 = []
    label = []
    user_id = []
    item_id = []
    data = {}
    file = file[(file['day_from_start'] >= START_FLAG) & \
                (file['day_from_start'] <= END_FLAG + 1)].copy()
    file.reset_index(drop=True, inplace=True)
    for index, row in file.iterrows():
        print('{}/{}'.format(index, len(file.index)))
        indexM1 = index - 1
        if index == 0:
            continue
        if file.ix[index]['user_id'] == file.ix[indexM1]['user_id'] and \
            file.ix[index]['item_id'] == file.ix[indexM1]['item_id']:
            continue
        else :
            subFile = file[indexTmp:index]
            # if len(subFile.index) < 20 :
            #     indexTmp = index
            #     continue
            user_id.append(file.ix[indexM1]['user_id'])
            item_id.append(file.ix[indexM1]['item_id'])
            dpTmp = subFile['day_from_start'].value_counts()
            feature1.append(dpTmp.get(START_FLAG,0))
            for i in range(START_FLAG+1, END_FLAG+1):
                feature1[-1] += dpTmp.get(i, 0)
            dpTmp = subFile[(subFile['day_from_start'] == END_FLAG) & \
                            ((subFile['operator'] == 2) | \
                            (subFile['operator'] == 3))]
            if len(dpTmp.index) != 0:
                feature2.append(1)
            else:
                feature2.append(0)
            if trainData:
                dpTmp = subFile[(subFile['day_from_start'] == LABLE_FLAG) & \
                                (subFile['operator'] == 4)]
                if len(dpTmp.index) != 0:
                    label.append(1)
                else:
                    label.append(0)
            indexTmp = index

    data['user_id'] = user_id
    data['item_id'] = item_id
    data['feature1'] = feature1
    data['feature2'] = feature2
    if trainData:
        data['lable'] = label
        dpTmp = pd.DataFrame(data, columns=['user_id', 'item_id', 'feature1', \
                                            'feature2', 'lable'])
        dpTmp.to_csv('data/fresh_comp_offline/6.txt', index=False)
    else:
        dpTmp = pd.DataFrame(data, columns=['user_id','item_id', 'feature1', \
                                            'feature2'])
        dpTmp.to_csv('data/fresh_comp_offline/7.txt', index=False)

    print '\n'

def features(data):
    """
    根据用户对item的操作,提取特征,并得到测试数据
    :param data: 特定[user_id,item_id]的所有操作类型和时间
    :return:
    """
    # 对一个item操作的时间间隔
    duration = data['time'].max() - data['time'].min()
    buyORNot = 0    # 是否购买(0,1)
    likeORNot = 0   # 是否收藏(0,1)
    cartORNot = 0   # 是否加入购物车(0,1)
    browseCount = 0 # 浏览次数,浏览操作代码为1,累加浏览
    No4 = True
    for index, row in data.iterrows():
        type = row['behavior_type']
        if type == 4 and No4:
            No4 = False
        if No4 and type == 1:
            continue

        if type == 1:
            browseCount += 1
        elif type == 2:
            likeORNot = 1
        elif type == 3:
            cartORNot = 1
        else:
            buyORNot = 1

    # 如果可以作为测试数据,那么作为测试数据返回
    if data['time'].min() < 36 and buyORNot == 0:
        return [], [duration, browseCount, likeORNot, cartORNot]
    # 否则,作为训练数据返回
    else:
        return [duration, browseCount, likeORNot, cartORNot, buyORNot], []

def getFeature():
    file = pd.read_csv('data/fresh_comp_offline/delWhoNOTBuy.csv')
    indexTmp = 0
    trainSet = np.array([0, 0, 0, 0])
    trainSetLable = np.array([0])
    testSet = np.array([0, 0, 0, 0, 0, 0])
    user_idTmp = file.ix[0]['user_id']
    item_idTmp = file.ix[0]['item_id']
    for index, row in file.iterrows():
        # same user
        if user_idTmp == row['user_id']:
            # same item
            if item_idTmp == row['item_id']:
                continue
            else: # different item
                # get the array of [user_id, item_id]
                train, test = features(file[indexTmp:index])

                if train != []:
                    trainSet = np.vstack((trainSet,
                                               np.array([train[:-1]])))
                    trainSetLable = np.vstack((trainSetLable,
                                                    np.array([train[-1:]])))
                if test != []:
                    test.insert(0, item_idTmp)
                    test.insert(0, user_idTmp)
                    testSet = np.vstack((testSet, np.array([test])))

                item_idTmp = row['item_id']
                indexTmp = index
        else: # different user
            # print '####user:{}'.format(user_idTmp)
            if trainSetLable[:].sum(axis=0) > 0 and len(testSet) != 1:
                logit = LogisticRegression(C=1.0)
                logit.fit(preprocessing.scale(trainSet.astype(float)),
                          trainSetLable.ravel())
                for i in range(len(testSet)):
                    # x = logit.predict(trainSet[0])
                    x = logit.predict(preprocessing.scale(testSet[i][
                                                          2:].astype(float)))
                    if x == 1:
                        print '{},{}'.format(int(testSet[i][0:1]),\
                                             int(testSet[i][1:2]))
            trainSet = np.array([0, 0, 0, 0])
            trainSetLable = np.array([0])
            testSet = np.array([[0, 0, 0, 0, 0, 0]])

            f = features(file[indexTmp:index])
            user_idTmp = row['user_id']
            item_idTmp = row['item_id']
            indexTmp = index

        # print file.ix[index-1]['user_id'], file.ix[index-1]['item_id'], f

    print '\n'

if __name__ == '__main__':
    # fuc1()
    # fuc2()
    # fuc3()
    # fuc4(28, 30)
    # fuc4(29, 31, False)
    getFeature()