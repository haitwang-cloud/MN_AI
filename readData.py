import numpy as np
import time
import pandas as pd

# 读取数据
part_1 = pd.read_csv('./dataset/round1_data1.txt', sep='$')
part_2 = pd.read_csv('./dataset/round1_data2.txt', sep='$')
part_1_2 = pd.concat([part_1, part_2])
# 按照名字排序，并初始化索引
part_1_2 = pd.DataFrame(part_1_2).sort_values('vid').reset_index(drop=True)
'''
    a = part_1_2[part_1_2.vid =='000330ad1f424114719b7525f400660b' ]
    000330ad1f424114719b7525f400660b
    part_1_2[part_1_2.vid =='000330ad1f424114719b7525f400660b'][part_1_2.table_id == '0102']
'''
begin_time = time.time()
print('begin')

# 数据简单处理
print('find_is_copy')
print(part_1_2.shape)
#
is_happen = part_1_2.groupby(['vid', 'table_id']).size().reset_index()
'''
    把数量大于1的提取出来
    b = is_happen[is_happen[is_happen.columns[2]]>1].reset_index(drop=True)
    然后去重
    len(b.vid.unique())         55633/57298
    也就是说，绝大数的都是有重复的，那么如何处理有两条以上的是一个问题
    那么现在需要考虑的是，重复是犹豫什么原因造成的，是不是同一table_id都有相同的重复，若是有，可能直接合并在一起，然后挖掘就可以
    如果不是的话，怎么处理还是一个问题
    b[b.table_id == '0102']
    到这一步就看出来了，长度还会有不同,该怎么处理好呢？
'''
# 重塑index用来去重
'''
    先把所有大于1的都给筛选出来
    size大于1的都为unique_part
    size等于1的都为no_unique_part
'''
is_happen['new_index'] = is_happen['vid'] + '_' + is_happen['table_id']
is_happen_new = is_happen[is_happen[0] > 1]['new_index']

part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['table_id']

unique_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]
unique_part = unique_part.sort_values(['vid', 'table_id'])
no_unique_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]
print('begin')
'''
    把重复的数据拼接在一起
    merge_table是用于处理数据数据的函数
    然后把这个前面等于1的与现在用过的合并在一起

'''


# 重复数据的拼接操作
def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df


part_1_2_not_unique = unique_part.groupby(['vid', 'table_id']).apply(merge_table).reset_index()
part_1_2_not_unique.rename(columns={0: 'field_results'}, inplace=True)
print('xxx')
tmp = pd.concat([part_1_2_not_unique, no_unique_part[['vid', 'table_id', 'field_results']]])
# 行列转换
print('finish')
tmp = tmp.pivot(index='vid', values='field_results', columns='table_id')
tmp.to_csv('./dataset/data_clean.csv')
print(tmp.shape)
print('totle time', time.time() - begin_time)