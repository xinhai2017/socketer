import numpy as np

radar_dict = {0: '一', 1: '丁', 2: '七', 3: '万',5: '三', 4: '丈',  7: '下', 8: '不', 6: '上', 9: '与', 10: '专',
              11: '两', 12: '义'}

# print(sorted(radar_dict.items(),key = lambda x:x[1],reverse = True))

# def numeric_return(oldarr):
#     dict={}
#     for i in range(oldarr.__le__()):
#         dict[oldarr[i]]=radar_dict[i] #创建字典
#     print(dict)
#     dicts=sorted(dict.items(), key=lambda x: x[1], reverse=True)
#     print(dicts)
#     list=dicts.values()
#     print(list)
#     return np.array(list)

prediction=[[7.9313677e-04,1.0686866e-18 ,4.4452804e-08 ,7.8642270e-06 ,3.8928491e-15
  ,1.4706162e-05, 5.7927178e-07, 6.4422062e-04, 9.9853909e-01, 2.7402631e-11,
  2.5275680e-13 ,1.8968781e-07, 7.6243133e-08]]
prediction=list(prediction)
print(prediction)
arrs=[]
for x in  range(10):
    max_index = np.argmax(prediction[0])
    print(radar_dict[max_index])
    arrs.append(radar_dict[max_index])
    prediction[0][max_index]=0

print(np.array(arrs))

