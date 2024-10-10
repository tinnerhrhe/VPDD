import numpy as np
def softmax(x):
    """ softmax function """
    
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行
    
    x -= np.max(x, axis = 0, keepdims = True) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    
    x = np.exp(x) / np.sum(np.exp(x), axis = 0, keepdims = True)
    
    return x

robot_datas = [np.load(f'./data/robot_latents_v1_{idr}.npz')['robot'] for idr in range(540)]
cnt_num = np.zeros(2048)
for path in robot_datas:
    for item in path:
        tmp = item.flatten()
        for d in tmp:
            cnt_num[d] += 1
import pdb;pdb.set_trace()
cnt_num = cnt_num / cnt_num.max()
print(cnt_num)
np.save('./data/cnt_num.npy', cnt_num)