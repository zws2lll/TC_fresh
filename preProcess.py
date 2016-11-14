#coding=utf-8
from datetime import datetime
import pandas as pd
import numpy as np

def changeDateToIntAndSort():
    """
    把时间格式转换成离2014-12-19日的距离,如把2014-12-18 06变成18
    :return: null
    """
    context = pd.read_csv('data/fresh_comp_offline/tianchi_fresh_comp_train_user.csv')
    # context = pd.read_csv('data/fresh_comp_offline/test.csv')

    LastDay = int(datetime.strptime('2014-12-19', '%Y-%m-%d').strftime('%j'))

    context['time'] = \
        context['time'].map(lambda x:
        24 * (LastDay - int(datetime.strptime(x[:10], '%Y-%m-%d').strftime(
            '%j'))) - int(x[-2:]))

    context.sort_index(by=['user_id', 'item_id', 'time',
                           'behavior_type'], ascending=True).\
        to_csv('data/fresh_comp_offline/changeDateToIntAndSort.csv',index=False,
               columns=['user_id', 'item_id', 'time', 'behavior_type'])

def removeWhoNOTBuy():
    """
    删除够买次数少于n的用户
    :return:
    """
    context = pd.read_csv('data/fresh_comp_offline/changeDateToIntAndSort.csv')
    # context.loc[len(context.index)] = np.array([0, 0, 0, 0])
    indexTmp = 0
    user_idTmp = context.ix[0]['user_id']
    delORNot = True
    delList = []
    length = len(context.index)

    for index, row in context.iterrows():
        print '{}/{}'.format(index, length)
        if user_idTmp != row['user_id']:
            if delORNot:
                delList.extend(range(indexTmp, index))
            indexTmp = index
            delORNot = True
            if row['behavior_type'] == 4:
                delORNot = False
            user_idTmp = row['user_id']
        elif delORNot == False:
            continue
        elif row['behavior_type'] == 4 and delORNot == True:
                delORNot = False
    if delORNot:
        delList.extend(range(indexTmp, len(context.index)))
    context.drop(delList, axis=0, inplace=True)
    context.to_csv('data/fresh_comp_offline/delWhoNOTBuy.csv',index=False)

if __name__ == '__main__':
    # changeDateToIntAndSort()
    removeWhoNOTBuy()