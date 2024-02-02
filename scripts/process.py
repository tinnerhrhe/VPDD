import numpy as np
file = open('/mnt/data/optimal/hehaoran/video_diff/eval_meta_301.txt','r')
file_data = file.readlines()
suc=[]
cut =0
#tasks = ["handle-press-v2","coffee-button-v2","button-press-v2"]
tasks = ["handle-press-v2","button-press-v2"]
for row in file_data:
    rate=float(row.split(',')[1][13:])
    task = str(row.split(',')[0][5:])
    if task in tasks:
        suc.append(rate)
suc=np.array(suc)
print(len(suc),suc.mean(),suc.std())
#print(f'success tasks:{cut}')
#R3M+Discrete Diffusion:(48.61333333333334, 1.1979241304115293)
#demo 1:(25.599999999999998, 0.989949493661166)
#demo 5:(46.4, 0.8286535263104032)
#demo 10:(50.96666666666666, 0.6236095644623235)
#demo 15:(0.5359999999999999, 0.56)
#corner_3 r3m:(15.200000000000001, 1.1313708498984762)
#dt:(30.266666666666666, 0.47842333648024327)
#soda:45.90,0.67