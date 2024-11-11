# Genetic algorithm-based production scheduling

## 개요

2022학년도 2학기 경영과학설계의 과제물로 제작한 자료로, genetic algorithm (GA)을 이용하여 부품 제조 공장의 생산 계획을 스케쥴링하는 프로그램을 제작하는 프로젝트입니다.

## 팀원

금종연, 김주은, 김진우, AYDIN ALI

## 코드

```python
import pandas as pd
import numpy as np
import datetime
import copy
import numpy
import random

from collections import deque
```

    C:\Users\KIMJINU\anaconda3\lib\site-packages\numpy\_distributor_init.py:30: UserWarning: loaded more than 1 DLL from .libs:
    C:\Users\KIMJINU\anaconda3\lib\site-packages\numpy\.libs\libopenblas.EL2C6PLE4ZYW3ECEVIV3OXXGRN2NRFM2.gfortran-win_amd64.dll
    C:\Users\KIMJINU\anaconda3\lib\site-packages\numpy\.libs\libopenblas.WCDJNK7YVMPZQ2ME2ZZHJJRJ3JIKNDB7.gfortran-win_amd64.dll
      warnings.warn("loaded more than 1 DLL from .libs:"



```python
pt_tmp=pd.read_excel("JSP_dataset_case.xlsx",sheet_name="Processing Time",index_col =[0])
ms_tmp=pd.read_excel("JSP_dataset_case.xlsx",sheet_name="Machines Sequence",index_col =[0])

start_time = datetime.datetime.strptime("2022-01-01 14:44", "%Y-%m-%d %H:%M")
moving_time = 0.1
Capa_utilization_rate =  0.715 
max_process=len(pt_tmp.columns)
num_job=len(pt_tmp)
#num_mc=7
L_num = 8
M_num = 3
R_num = 2
G_num = 5
CNC_num = 1
MCT_num = 7
B_num = 7
num_mc = L_num+M_num+R_num+G_num+CNC_num+MCT_num+B_num
num_gene=max_process*num_job - pt_tmp.isnull().sum().sum() 

pt=[list(map(float,pt_tmp.iloc[i][0:max_process-pt_tmp.iloc[i].isnull().sum()])) for i in range(num_job)]
ms=[list(map(float,ms_tmp.iloc[i][0:max_process-ms_tmp.iloc[i].isnull().sum()])) for i in range(num_job)]

population_size = 50
crossover_rate = 0.8
mutation_rate = 0.8
mutation_selection_rate = 0.5
num_mutation_jobs = round(num_gene*mutation_selection_rate)
generation = 1000


Tbest=999999999999999
current_time = datetime.datetime.now()
```


```python
# LPT
# Data
data = []
job_flow = []
LPT_IP = []

for i in range(len(pt)):
    data.append([])
    for j in range(len(pt[i])) : 
        if j == 0 :
            past_job_is_done = True
        else:
            past_job_is_done = False
        data[i].append([pt[i][j],ms[i][j],i,j,past_job_is_done])
        job_flow.append([pt[i][j],ms[i][j],i,j,past_job_is_done])

job_flow_sorted = sorted(job_flow, key = lambda x : (x[4], x[0])) # LPT 정렬기준

L_job = []
L_done = []

for i in range(L_num) :
    L_job.append(deque([]))
    L_done.append(dict(task = "L-"+ str(i+1), last_time = start_time, process = []))
    
    
M_job = []
M_done = []
for i in range(M_num) :
    M_job.append(deque([]))
    M_done.append(dict(task = "M-"+ str(i+1), last_time = start_time, process = []))

R_job = []
R_done = []
for i in range(R_num) :
    R_job.append(deque([]))
    R_done.append(dict(task = "R-"+ str(i+1), last_time = start_time, process = []))
    

G_job = []
G_done = []
for i in range(G_num) :
    G_job.append(deque([]))
    G_done.append(dict(task = "G-"+ str(i+1), last_time = start_time, process = []))

CNC_job = []
CNC_done = []
for i in range(CNC_num) :
    CNC_job.append(deque([]))
    CNC_done.append(dict(task = "CNC-"+ str(i+1), last_time = start_time, process = []))
    
MCT_job = []
MCT_done = []
for i in range(MCT_num) :
    MCT_job.append(deque([]))
    MCT_done.append(dict(task = "MCT-"+ str(i+1), last_time = start_time, process = []))

B_job = []
B_done = []
for i in range(B_num) :
    B_job.append(deque([]))
    B_done.append(dict(task = "B-"+ str(i+1), last_time = start_time, process = []))

jobs_done = [L_done] + [M_done] + [R_done] + [G_done] + [CNC_done] + [MCT_done] + [B_done]

process_done = pd.DataFrame([],columns=['Machine', 'Start', 'Finish'])
process_done['Start'] = pd.to_datetime(process_done['Start'])
process_done['Finish'] = pd.to_datetime(process_done['Finish'])

# [0.8, 2.0, 26, 0, True]
process_done = []

while job_flow_sorted :
    current_job = job_flow_sorted.pop()
    for job in job_flow_sorted : 
        if (job[2] == current_job[2]) and (job[3]==current_job[3]+1) : 
            job[4] = True
    current_process = current_job[1]
    min_time = datetime.datetime.strptime("2030-01-01 14:44", "%Y-%m-%d %H:%M")
    done_time = datetime.datetime.strptime("2022-01-01 14:44", "%Y-%m-%d %H:%M")
    min_task_id = -1
    # task의 마지막 done time이 같을 경우 id가 제일 빠른 machine을 채택한다.
    current_machine = jobs_done[int(current_process)-1]

    # process의 앞 단계가 끝나지 않았는데 job에 들어가면 안되기 때문에 해당 logic이 필요함.
    all_more_done = True
    if process_done :
        for process in process_done:
                if process['location'] == (current_job[2], current_job[3]-1):
                    done_time = process['Finish']
                    break

        for i, task in enumerate(current_machine) :
            # 여러 개의 task가 done_time보다 같거나 일찍 끝나면 위에서 부터 아무거나 선택
            if (task['last_time'] <= done_time) :
                min_time = done_time
                min_task_id = i
                all_more_done = False
                break

        if all_more_done :
            for i, task in enumerate(current_machine) :
                ## 이런 조건이 없을 경우에 그냥 last_time이 done_time보다 큰 것 아무거나 골라서 job에 할당한다
                if task['last_time'] < min_time :
                    min_time = task['last_time']
                    min_task_id = i

    else : 
        for i, task in enumerate(current_machine) :
            if task['last_time'] <= min_time :
                min_time = task['last_time']
                min_task_id = i



    
    if current_machine[min_task_id]['last_time'] == datetime.datetime.strptime("2022-01-01 14:44", "%Y-%m-%d %H:%M"):
       current_machine[min_task_id]['last_time'] += datetime.timedelta(hours = current_job[0] / Capa_utilization_rate)
       current_machine[min_task_id]['process'].append((current_job[2], current_job[3]))
       process_done.append(dict(Machine = current_machine[min_task_id]['task'], Start = min_time, \
        Finish = current_machine[min_task_id]['last_time'], location = (current_job[2], current_job[3])))
    
    else : 
        rest_time = min_time - current_machine[min_task_id]['last_time']
        if rest_time <= datetime.timedelta(hours = moving_time):
            current_machine[min_task_id]['last_time'] = min_time + datetime.timedelta(hours = moving_time) - rest_time + datetime.timedelta(hours = current_job[0] / Capa_utilization_rate)
            current_machine[min_task_id]['process'].append((current_job[2], current_job[3]))
            process_done.append(dict(Machine = current_machine[min_task_id]['task'], Start = min_time + datetime.timedelta(hours = moving_time), \
                Finish = current_machine[min_task_id]['last_time'], location = (current_job[2], current_job[3])))
        else : 
            current_machine[min_task_id]['last_time'] = min_time + datetime.timedelta(hours = current_job[0] / Capa_utilization_rate)
            current_machine[min_task_id]['process'].append((current_job[2], current_job[3]))
            process_done.append(dict(Machine = current_machine[min_task_id]['task'], Start = min_time + datetime.timedelta(hours = moving_time), \
                Finish = current_machine[min_task_id]['last_time'], location = (current_job[2], current_job[3])))
    
    job_flow_sorted = sorted(job_flow_sorted, key = lambda x : (x[4], x[0]))
    
df = pd.DataFrame( [d for d in process_done])

for i in range(len(df)):
    LPT_IP.append(df.iloc[i]['location'][0])
```


```python
j_keys=[j for j in range(num_job)]
key_count={key:0 for key in j_keys}
j_count={key:0 for key in j_keys}
L_keys=["1"+"_"+str(j+1) for j in range(L_num)]
M_keys=["2"+"_"+str(j+1) for j in range(M_num)]
R_keys=["3"+"_"+str(j+1) for j in range(R_num)]
G_keys=["4"+"_"+str(j+1) for j in range(G_num)]
CNC_keys=["5"+"_"+str(j+1) for j in range(CNC_num)]
MCT_keys=["6"+"_"+str(j+1) for j in range(MCT_num)]
B_keys=["7"+"_"+str(j+1) for j in range(B_num)]
m_keys = L_keys+M_keys+R_keys+G_keys+CNC_keys+MCT_keys+B_keys
m_count={key:0 for key in m_keys}

for i in LPT_IP:
    gen_t=float(pt[i][key_count[i]])/Capa_utilization_rate
    gen_m=int(ms[i][key_count[i]])
    j_count[i]=j_count[i]+gen_t+moving_time
    L_min = 9999999
    M_min = 9999999
    R_min = 9999999
    G_min = 9999999
    CNC_min = 9999999
    MCT_min = 9999999
    B_min = 9999999

    if gen_m == 1:
        for k in range(1,L_num+1):
            L_temp = m_count["1_"+str(k)]
            if L_temp < L_min:
                L_min = L_temp
                index = "1_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 2:
        for k in range(1,M_num+1):
            M_temp = m_count["2_"+str(k)]
            if M_temp < M_min:
                M_min = M_temp
                index = "2_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 3:
        for k in range(1,R_num+1):
            R_temp = m_count["3_"+str(k)]
            if R_temp < R_min:
                R_min = R_temp
                index = "3_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 4:
        for k in range(1,G_num+1):
            G_temp = m_count["4_"+str(k)]
            if G_temp < G_min:
                G_min = G_temp
                index = "4_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 5:
        for k in range(1,CNC_num+1):
            CNC_temp = m_count["5_"+str(k)]
            if CNC_temp < CNC_min:
                CNC_min = CNC_temp
                index = "5_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 6:
        for k in range(1,MCT_num+1):
            MCT_temp = m_count["6_"+str(k)]
            if MCT_temp < MCT_min:
                MCT_min = MCT_temp
                index = "6_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    else:
        for k in range(1,B_num+1):
            B_temp = m_count["7_"+str(k)]
            if B_temp < B_min:
                B_min = B_temp
                index = "7_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    if m_count[index]<j_count[i]:
        m_count[index]=j_count[i]
    elif m_count[index]>j_count[i]:
        j_count[i]=m_count[index]

    key_count[i]=key_count[i]+1

makespan_LPT=max(j_count.values())
```


```python
print("optimal value:%f"%makespan_LPT)
print('End time:%s'%(str(start_time + datetime.timedelta(hours=makespan_LPT))))
```

    optimal value:986.613986
    End time:2022-02-11 17:20:50.349650



```python
# SPT
# Data
data = []
job_flow = []
SPT_IP = []

for i in range(len(pt)):
    data.append([])
    for j in range(len(pt[i])) : 
        if j == 0 :
            past_job_is_done = True
        else:
            past_job_is_done = False
        data[i].append([pt[i][j],ms[i][j],i,j,past_job_is_done])
        job_flow.append([pt[i][j],ms[i][j],i,j,past_job_is_done])

job_flow_sorted = sorted(job_flow, key = lambda x : (x[4], -x[0])) # SPT 정렬기준

L_job = []
L_done = []

for i in range(L_num) :
    L_job.append(deque([]))
    L_done.append(dict(task = "L-"+ str(i+1), last_time = start_time, process = []))
    
    
M_job = []
M_done = []
for i in range(M_num) :
    M_job.append(deque([]))
    M_done.append(dict(task = "M-"+ str(i+1), last_time = start_time, process = []))

R_job = []
R_done = []
for i in range(R_num) :
    R_job.append(deque([]))
    R_done.append(dict(task = "R-"+ str(i+1), last_time = start_time, process = []))
    

G_job = []
G_done = []
for i in range(G_num) :
    G_job.append(deque([]))
    G_done.append(dict(task = "G-"+ str(i+1), last_time = start_time, process = []))

CNC_job = []
CNC_done = []
for i in range(CNC_num) :
    CNC_job.append(deque([]))
    CNC_done.append(dict(task = "CNC-"+ str(i+1), last_time = start_time, process = []))
    
MCT_job = []
MCT_done = []
for i in range(MCT_num) :
    MCT_job.append(deque([]))
    MCT_done.append(dict(task = "MCT-"+ str(i+1), last_time = start_time, process = []))

B_job = []
B_done = []
for i in range(B_num) :
    B_job.append(deque([]))
    B_done.append(dict(task = "B-"+ str(i+1), last_time = start_time, process = []))

jobs_done = [L_done] + [M_done] + [R_done] + [G_done] + [CNC_done] + [MCT_done] + [B_done]

process_done = pd.DataFrame([],columns=['Machine', 'Start', 'Finish'])
process_done['Start'] = pd.to_datetime(process_done['Start'])
process_done['Finish'] = pd.to_datetime(process_done['Finish'])

# [0.8, 2.0, 26, 0, True]
process_done = []

while job_flow_sorted :
    current_job = job_flow_sorted.pop()
    for job in job_flow_sorted : 
        if (job[2] == current_job[2]) and (job[3]==current_job[3]+1) : 
            job[4] = True
    current_process = current_job[1]
    min_time = datetime.datetime.strptime("2030-01-01 14:44", "%Y-%m-%d %H:%M")
    done_time = datetime.datetime.strptime("2022-01-01 14:44", "%Y-%m-%d %H:%M")
    min_task_id = -1
    # task의 마지막 done time이 같을 경우 id가 제일 빠른 machine을 채택한다.
    current_machine = jobs_done[int(current_process)-1]

    # process의 앞 단계가 끝나지 않았는데 job에 들어가면 안되기 때문에 해당 logic이 필요함.
    all_more_done = True
    if process_done :
        for process in process_done:
                if process['location'] == (current_job[2], current_job[3]-1):
                    done_time = process['Finish']
                    break

        for i, task in enumerate(current_machine) :
            # 여러 개의 task가 done_time보다 같거나 일찍 끝나면 위에서 부터 아무거나 선택
            if (task['last_time'] <= done_time) :
                min_time = done_time
                min_task_id = i
                all_more_done = False
                break

        if all_more_done :
            for i, task in enumerate(current_machine) :
                ## 이런 조건이 없을 경우에 그냥 last_time이 done_time보다 큰 것 아무거나 골라서 job에 할당한다
                if task['last_time'] < min_time :
                    min_time = task['last_time']
                    min_task_id = i

    else : 
        for i, task in enumerate(current_machine) :
            if task['last_time'] <= min_time :
                min_time = task['last_time']
                min_task_id = i



    
    if current_machine[min_task_id]['last_time'] == datetime.datetime.strptime("2022-01-01 14:44", "%Y-%m-%d %H:%M"):
       current_machine[min_task_id]['last_time'] += datetime.timedelta(hours = current_job[0] / Capa_utilization_rate)
       current_machine[min_task_id]['process'].append((current_job[2], current_job[3]))
       process_done.append(dict(Machine = current_machine[min_task_id]['task'], Start = min_time, \
        Finish = current_machine[min_task_id]['last_time'], location = (current_job[2], current_job[3])))
    
    else : 
        rest_time = min_time - current_machine[min_task_id]['last_time']
        if rest_time <= datetime.timedelta(hours = moving_time):
            current_machine[min_task_id]['last_time'] = min_time + datetime.timedelta(hours = moving_time) - rest_time + datetime.timedelta(hours = current_job[0] / Capa_utilization_rate)
            current_machine[min_task_id]['process'].append((current_job[2], current_job[3]))
            process_done.append(dict(Machine = current_machine[min_task_id]['task'], Start = min_time + datetime.timedelta(hours = moving_time), \
                Finish = current_machine[min_task_id]['last_time'], location = (current_job[2], current_job[3])))
        else : 
            current_machine[min_task_id]['last_time'] = min_time + datetime.timedelta(hours = current_job[0] / Capa_utilization_rate)
            current_machine[min_task_id]['process'].append((current_job[2], current_job[3]))
            process_done.append(dict(Machine = current_machine[min_task_id]['task'], Start = min_time + datetime.timedelta(hours = moving_time), \
                Finish = current_machine[min_task_id]['last_time'], location = (current_job[2], current_job[3])))
    
    job_flow_sorted = sorted(job_flow_sorted, key = lambda x : (x[4], -x[0]))
    
df = pd.DataFrame( [d for d in process_done])

for i in range(len(df)):
    SPT_IP.append(df.iloc[i]['location'][0])
```


```python
j_keys=[j for j in range(num_job)]
key_count={key:0 for key in j_keys}
j_count={key:0 for key in j_keys}
L_keys=["1"+"_"+str(j+1) for j in range(L_num)]
M_keys=["2"+"_"+str(j+1) for j in range(M_num)]
R_keys=["3"+"_"+str(j+1) for j in range(R_num)]
G_keys=["4"+"_"+str(j+1) for j in range(G_num)]
CNC_keys=["5"+"_"+str(j+1) for j in range(CNC_num)]
MCT_keys=["6"+"_"+str(j+1) for j in range(MCT_num)]
B_keys=["7"+"_"+str(j+1) for j in range(B_num)]
m_keys = L_keys+M_keys+R_keys+G_keys+CNC_keys+MCT_keys+B_keys
m_count={key:0 for key in m_keys}

for i in SPT_IP:
    gen_t=float(pt[i][key_count[i]])/Capa_utilization_rate
    gen_m=int(ms[i][key_count[i]])
    j_count[i]=j_count[i]+gen_t+moving_time
    L_min = 9999999
    M_min = 9999999
    R_min = 9999999
    G_min = 9999999
    CNC_min = 9999999
    MCT_min = 9999999
    B_min = 9999999

    if gen_m == 1:
        for k in range(1,L_num+1):
            L_temp = m_count["1_"+str(k)]
            if L_temp < L_min:
                L_min = L_temp
                index = "1_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 2:
        for k in range(1,M_num+1):
            M_temp = m_count["2_"+str(k)]
            if M_temp < M_min:
                M_min = M_temp
                index = "2_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 3:
        for k in range(1,R_num+1):
            R_temp = m_count["3_"+str(k)]
            if R_temp < R_min:
                R_min = R_temp
                index = "3_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 4:
        for k in range(1,G_num+1):
            G_temp = m_count["4_"+str(k)]
            if G_temp < G_min:
                G_min = G_temp
                index = "4_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 5:
        for k in range(1,CNC_num+1):
            CNC_temp = m_count["5_"+str(k)]
            if CNC_temp < CNC_min:
                CNC_min = CNC_temp
                index = "5_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 6:
        for k in range(1,MCT_num+1):
            MCT_temp = m_count["6_"+str(k)]
            if MCT_temp < MCT_min:
                MCT_min = MCT_temp
                index = "6_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    else:
        for k in range(1,B_num+1):
            B_temp = m_count["7_"+str(k)]
            if B_temp < B_min:
                B_min = B_temp
                index = "7_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    if m_count[index]<j_count[i]:
        m_count[index]=j_count[i]
    elif m_count[index]>j_count[i]:
        j_count[i]=m_count[index]

    key_count[i]=key_count[i]+1

makespan_SPT=max(j_count.values())
```


```python
print("optimal value:%f"%makespan_SPT)
print('End time:%s'%(str(start_time + datetime.timedelta(hours=makespan_SPT))))
```

    optimal value:1098.423077
    End time:2022-02-16 09:09:23.076923



```python
# GA
best_list,best_obj=[],[]
population_list=[]
makespan_record=[]
for i in range(int(population_size*0.8)):
    nxm_random_num=list(np.random.permutation(num_gene)) 
    population_list.append(nxm_random_num) 
    for j in range(num_gene):
        population_list[i][j]=population_list[i][j]%num_job 
        
for i in range(int(population_size*0.1)):
    population_list.append(LPT_IP)
    population_list.append(SPT_IP)
```


```python
for generation_num in range(1,generation+1):
    print(generation_num)
    Tbest_now=99999999999         
    
    for m in range(population_size):
        job_count={}
        larger,less=[],[]
        for i in range(num_job):
            process_num_i = len(ms_tmp.iloc[i]) - ms_tmp.iloc[i].isnull().sum()
            if i in population_list[m]:
                count=population_list[m].count(i)
                pos=population_list[m].index(i)
                job_count[i]=[count,pos]
            else:
                count=0
                job_count[i]=[count,0]

            if count>process_num_i:
                larger.append(i)
            elif count<process_num_i:
                less.append(i)

        for k in range(len(larger)):
            chg_job=larger[k]
            process_num_chg_job = len(ms_tmp.iloc[chg_job]) - ms_tmp.iloc[chg_job].isnull().sum()
            while job_count[chg_job][0]>process_num_chg_job:
                for d in range(len(less)):

                    process_num_d = len(ms_tmp.iloc[less[d]]) - ms_tmp.iloc[less[d]].isnull().sum()
                    if job_count[less[d]][0]<process_num_d:       
                        population_list[m][job_count[chg_job][1]]=less[d]
                        job_count[chg_job][1]=population_list[m].index(chg_job)
                        job_count[chg_job][0]=job_count[chg_job][0]-1
                        job_count[less[d]][0]=job_count[less[d]][0]+1                    
                    if job_count[chg_job][0]==process_num_chg_job:
                        break

    parent_list=copy.deepcopy(population_list)
    offspring_list=copy.deepcopy(population_list)
    S=list(np.random.permutation(population_size))
    
    for m in range(int(population_size/2)):
        crossover_prob=np.random.rand()
        if crossover_rate>=crossover_prob:
            parent_1= population_list[S[2*m]][:]
            parent_2= population_list[S[2*m+1]][:]
            child_1=parent_1[:]
            child_2=parent_2[:]
            cutpoint=list(np.random.choice(num_gene, 2, replace=False))
            cutpoint.sort()

            child_1[cutpoint[0]:cutpoint[1]]=parent_2[cutpoint[0]:cutpoint[1]]
            child_2[cutpoint[0]:cutpoint[1]]=parent_1[cutpoint[0]:cutpoint[1]]
            offspring_list[S[2*m]]=child_1[:]
                        
                        
                        
    for m in range(population_size):
        job_count={}
        larger,less=[],[]
        for i in range(num_job):
            process_num_i = len(ms_tmp.iloc[i]) - ms_tmp.iloc[i].isnull().sum()
            if i in offspring_list[m]:
                count=offspring_list[m].count(i)
                pos=offspring_list[m].index(i)
                job_count[i]=[count,pos]
            else:
                count=0
                job_count[i]=[count,0]

            if count>process_num_i:
                larger.append(i)
            elif count<process_num_i:
                less.append(i)

        for k in range(len(larger)):
            chg_job=larger[k]
            process_num_chg_job = len(ms_tmp.iloc[chg_job]) - ms_tmp.iloc[chg_job].isnull().sum()
            while job_count[chg_job][0]>process_num_chg_job:
                for d in range(len(less)):
                    process_num_d = len(ms_tmp.iloc[less[d]]) - ms_tmp.iloc[less[d]].isnull().sum()
                    if job_count[less[d]][0]<process_num_d:       
                        offspring_list[m][job_count[chg_job][1]]=less[d]
                        job_count[chg_job][1]=offspring_list[m].index(chg_job)
                        job_count[chg_job][0]=job_count[chg_job][0]-1
                        job_count[less[d]][0]=job_count[less[d]][0]+1                    
                    if job_count[chg_job][0]==process_num_chg_job:
                        break

    for m in range(len(offspring_list)):
        mutation_prob=np.random.rand()
        if mutation_rate >= mutation_prob:
            m_chg=list(np.random.choice(num_gene, num_mutation_jobs, replace=False)) 
            t_value_last=offspring_list[m][m_chg[0]] 
            for i in range(num_mutation_jobs-1):
                offspring_list[m][m_chg[i]]=offspring_list[m][m_chg[i+1]]

            offspring_list[m][m_chg[num_mutation_jobs-1]]=t_value_last

    
    total_chromosome=copy.deepcopy(parent_list)+copy.deepcopy(offspring_list)
    chrom_fitness,chrom_fit=[],[]
    total_fitness=0
    for m in range(population_size*2):
        j_keys=[j for j in range(num_job)]
        key_count={key:0 for key in j_keys}
        j_count={key:0 for key in j_keys}
        L_keys=["1"+"_"+str(j+1) for j in range(L_num)]
        M_keys=["2"+"_"+str(j+1) for j in range(M_num)]
        R_keys=["3"+"_"+str(j+1) for j in range(R_num)]
        G_keys=["4"+"_"+str(j+1) for j in range(G_num)]
        CNC_keys=["5"+"_"+str(j+1) for j in range(CNC_num)]
        MCT_keys=["6"+"_"+str(j+1) for j in range(MCT_num)]
        B_keys=["7"+"_"+str(j+1) for j in range(B_num)]
        m_keys = L_keys+M_keys+R_keys+G_keys+CNC_keys+MCT_keys+B_keys
        m_count={key:0 for key in m_keys}

        for i in total_chromosome[m]:
            gen_t=float(pt[i][key_count[i]])/Capa_utilization_rate
            gen_m=int(ms[i][key_count[i]])
            j_count[i]=j_count[i]+gen_t+moving_time
            L_min = 9999999
            M_min = 9999999
            R_min = 9999999
            G_min = 9999999
            CNC_min = 9999999
            MCT_min = 9999999
            B_min = 9999999

            if gen_m == 1:
                for k in range(1,L_num+1):
                    L_temp = m_count["1_"+str(k)]
                    if L_temp < L_min:
                        L_min = L_temp
                        index = "1_"+str(k)
                    else:
                        continue
                m_count[index]=m_count[index]+gen_t

            elif gen_m == 2:
                for k in range(1,M_num+1):
                    M_temp = m_count["2_"+str(k)]
                    if M_temp < M_min:
                        M_min = M_temp
                        index = "2_"+str(k)
                    else:
                        continue
                m_count[index]=m_count[index]+gen_t

            elif gen_m == 3:
                for k in range(1,R_num+1):
                    R_temp = m_count["3_"+str(k)]
                    if R_temp < R_min:
                        R_min = R_temp
                        index = "3_"+str(k)
                    else:
                        continue
                m_count[index]=m_count[index]+gen_t

            elif gen_m == 4:
                for k in range(1,G_num+1):
                    G_temp = m_count["4_"+str(k)]
                    if G_temp < G_min:
                        G_min = G_temp
                        index = "4_"+str(k)
                    else:
                        continue
                m_count[index]=m_count[index]+gen_t

            elif gen_m == 5:
                for k in range(1,CNC_num+1):
                    CNC_temp = m_count["5_"+str(k)]
                    if CNC_temp < CNC_min:
                        CNC_min = CNC_temp
                        index = "5_"+str(k)
                    else:
                        continue
                m_count[index]=m_count[index]+gen_t

            elif gen_m == 6:
                for k in range(1,MCT_num+1):
                    MCT_temp = m_count["6_"+str(k)]
                    if MCT_temp < MCT_min:
                        MCT_min = MCT_temp
                        index = "6_"+str(k)
                    else:
                        continue
                m_count[index]=m_count[index]+gen_t

            else:
                for k in range(1,B_num+1):
                    B_temp = m_count["7_"+str(k)]
                    if B_temp < B_min:
                        B_min = B_temp
                        index = "7_"+str(k)
                    else:
                        continue
                m_count[index]=m_count[index]+gen_t

            if m_count[index]<j_count[i]:
                m_count[index]=j_count[i]
            elif m_count[index]>j_count[i]:
                j_count[i]=m_count[index]

            key_count[i]=key_count[i]+1

        makespan=max(j_count.values())
        chrom_fitness.append(1/makespan)
        chrom_fit.append(makespan)
        total_fitness=total_fitness+chrom_fitness[m]

    pk,qk=[],[]

    for i in range(population_size*2):
        pk.append(chrom_fitness[i]/total_fitness)
    for i in range(population_size*2):
        cumulative=0
        for j in range(0,i+1):
            cumulative=cumulative+pk[j]
        qk.append(cumulative)

    selection_rand=[np.random.rand() for i in range(population_size)]

    for i in range(population_size):
        if selection_rand[i]<=qk[0]:
            population_list[i]=copy.deepcopy(total_chromosome[0])
        else:
            for j in range(0,population_size*2-1):
                if selection_rand[i]>qk[j] and selection_rand[i]<=qk[j+1]:
                    population_list[i]=copy.deepcopy(total_chromosome[j+1])
                    break

    for i in range(population_size*2):
        if chrom_fit[i]<Tbest_now:
            Tbest_now=chrom_fit[i]
            sequence_now=copy.deepcopy(total_chromosome[i])

    if Tbest_now<=Tbest:
        Tbest=Tbest_now
        sequence_best=copy.deepcopy(sequence_now)

    makespan_record.append(Tbest)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10



```python
'''----------result----------'''
print("optimal sequence",sequence_best)
print("optimal value:%f"%Tbest)
print('End time:%s'%(str(start_time + datetime.timedelta(hours=Tbest))))
print('the elapsed time:%s'% (datetime.datetime.now() - current_time))
```

    optimal sequence [82, 15, 19, 101, 57, 35, 104, 8, 1, 34, 59, 57, 16, 55, 44, 36, 107, 80, 10, 40, 116, 75, 115, 84, 36, 6, 109, 32, 16, 64, 8, 96, 53, 27, 22, 14, 1, 53, 2, 70, 70, 61, 87, 105, 45, 25, 61, 73, 9, 104, 67, 71, 61, 77, 65, 54, 85, 113, 103, 26, 88, 94, 112, 100, 17, 59, 105, 15, 99, 88, 110, 17, 18, 107, 112, 54, 62, 41, 72, 24, 58, 109, 35, 82, 5, 80, 93, 115, 64, 115, 35, 110, 94, 116, 1, 13, 83, 24, 103, 12, 26, 37, 65, 98, 42, 116, 67, 27, 92, 44, 99, 20, 20, 85, 49, 56, 13, 111, 97, 99, 68, 77, 56, 73, 18, 49, 81, 91, 108, 99, 36, 79, 50, 106, 32, 23, 101, 25, 53, 55, 95, 21, 60, 116, 11, 117, 28, 31, 112, 24, 103, 108, 119, 32, 43, 21, 100, 42, 105, 111, 89, 37, 12, 94, 48, 97, 37, 72, 23, 58, 38, 90, 42, 22, 38, 74, 27, 76, 85, 22, 59, 101, 109, 83, 47, 33, 96, 77, 53, 69, 34, 78, 89, 21, 94, 60, 63, 71, 107, 13, 72, 72, 101, 102, 108, 5, 66, 12, 104, 54, 7, 82, 33, 86, 17, 11, 45, 76, 48, 76, 4, 43, 114, 105, 115, 79, 51, 64, 81, 113, 62, 35, 93, 71, 63, 24, 93, 61, 46, 87, 77, 81, 76, 52, 63, 92, 28, 112, 78, 14, 118, 102, 41, 75, 74, 103, 37, 18, 5, 21, 64, 114, 68, 96, 40, 14, 69, 13, 95, 3, 53, 112, 108, 69, 55, 110, 45, 44, 74, 5, 0, 114, 41, 7, 25, 52, 2, 84, 58, 14, 56, 0, 3, 106, 73, 46, 28, 57, 101, 84, 112, 4, 81, 111, 49, 98, 117, 72, 114, 82, 20, 65, 39, 104, 118, 32, 13, 48, 80, 32, 75, 63, 2, 24, 50, 40, 48, 102, 66, 89, 30, 22, 117, 100, 104, 8, 67, 93, 100, 99, 88, 59, 55, 4, 97, 59, 86, 25, 67, 33, 68, 36, 19, 2, 8, 36, 15, 9, 68, 42, 31, 88, 95, 61, 23, 62, 21, 25, 60, 47, 75, 39, 38, 4, 117, 19, 54, 94, 31, 44, 65, 96, 107, 27, 76, 116, 34, 119, 102, 20, 43, 16, 10, 74, 93, 3, 0, 95, 61, 54, 28, 30, 101, 14, 41, 17, 29, 34, 57, 23, 118, 91, 15, 90, 51, 47, 97, 56, 32, 62, 21, 47, 9, 87, 19, 83, 92, 29, 6, 1, 72, 113, 7, 60, 84, 52, 45, 70, 98, 64, 65, 7, 78, 87, 30, 19, 29, 105, 16]
    optimal value:986.613986
    End time:2022-02-11 17:20:50.349650
    the elapsed time:0:00:45.351591



```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot([i for i in range(len(makespan_record))],makespan_record,'b')
plt.ylabel('makespan',fontsize=15)
plt.xlabel('generation',fontsize=15)
#plt.show()
plt.tight_layout()
plt.savefig('Makespan_fig.jpg')
```


​    
![png](output_11_0.png)
​    



```python
import pandas as pd
import chart_studio.plotly as py
import plotly
import plotly.figure_factory

j_keys=[j for j in range(num_job)]
key_count={key:0 for key in j_keys}
j_count={key:0 for key in j_keys}
L_keys=["1"+"_"+str(j+1) for j in range(L_num)]
M_keys=["2"+"_"+str(j+1) for j in range(M_num)]
R_keys=["3"+"_"+str(j+1) for j in range(R_num)]
G_keys=["4"+"_"+str(j+1) for j in range(G_num)]
CNC_keys=["5"+"_"+str(j+1) for j in range(CNC_num)]
MCT_keys=["6"+"_"+str(j+1) for j in range(MCT_num)]
B_keys=["7"+"_"+str(j+1) for j in range(B_num)]
m_keys = L_keys+M_keys+R_keys+G_keys+CNC_keys+MCT_keys+B_keys
m_count={key:0 for key in m_keys}
j_record={}
j_record_2={}

for i in sequence_best:
    gen_t=float(pt[i][key_count[i]])/Capa_utilization_rate
    gen_m=int(ms[i][key_count[i]])
    j_count[i]=j_count[i]+gen_t+moving_time
    L_min = 9999999
    M_min = 9999999
    R_min = 9999999
    G_min = 9999999
    CNC_min = 9999999
    MCT_min = 9999999
    B_min = 9999999

    if gen_m == 1:
        for k in range(1,L_num+1):
            L_temp = m_count["1_"+str(k)]
            if L_temp < L_min:
                L_min = L_temp
                index = "1_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 2:
        for k in range(1,M_num+1):
            M_temp = m_count["2_"+str(k)]
            if M_temp < M_min:
                M_min = M_temp
                index = "2_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 3:
        for k in range(1,R_num+1):
            R_temp = m_count["3_"+str(k)]
            if R_temp < R_min:
                R_min = R_temp
                index = "3_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 4:
        for k in range(1,G_num+1):
            G_temp = m_count["4_"+str(k)]
            if G_temp < G_min:
                G_min = G_temp
                index = "4_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 5:
        for k in range(1,CNC_num+1):
            CNC_temp = m_count["5_"+str(k)]
            if CNC_temp < CNC_min:
                CNC_min = CNC_temp
                index = "5_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    elif gen_m == 6:
        for k in range(1,MCT_num+1):
            MCT_temp = m_count["6_"+str(k)]
            if MCT_temp < MCT_min:
                MCT_min = MCT_temp
                index = "6_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t

    else:
        for k in range(1,B_num+1):
            B_temp = m_count["7_"+str(k)]
            if B_temp < B_min:
                B_min = B_temp
                index = "7_"+str(k)
            else:
                continue
        m_count[index]=m_count[index]+gen_t
    
    if m_count[index]<j_count[i]:
        m_count[index]=j_count[i]
    elif m_count[index]>j_count[i]:
        j_count[i]=m_count[index]
    
    current_time=datetime.timedelta(hours=j_count[i]-(pt[i][key_count[i]]/Capa_utilization_rate)) # convert seconds to hours, minutes and seconds
    end_time=datetime.timedelta(hours=j_count[i])
    try:
        j_record[(i,index)]
        j_record_2[(i,index)]=[current_time,end_time]
    except:
        j_record[(i,index)]=[current_time,end_time]
    #print(gen_m)
    
    key_count[i]=key_count[i]+1

df=[]

for (j,m) in j_record.keys():
    df.append(dict(Task='Machine %s'%(m), Start='%s'%(str(start_time + j_record[(j,m)][0])), Finish='%s'%(str(start_time + j_record[(j,m)][1])),Resource='Job %s'%(j+1)))

for (j,m) in j_record_2.keys():
    df.append(dict(Task='Machine %s'%(m), Start='%s'%(str(start_time + j_record_2[(j,m)][0])), Finish='%s'%(str(start_time + j_record_2[(j,m)][1])),Resource='Job %s'%(j+1)))
    
r = lambda: random.randint(0,255)            
colors = ['#%02X%02X%02X' % (r(),r(),r())]
for i in range(1, num_job):                                   
    colors.append('#%02X%02X%02X' % (r(),r(),r()))
    
    
    
fig = plotly.figure_factory.create_gantt(df, colors=colors,index_col='Resource', show_colorbar=True, group_tasks=True, showgrid_x=True, title='Job shop Schedule')
fig.show()
plotly.io.write_image(fig,file="scheduling_fig.jpg", format="jpeg",scale=None, width=None, height=None)
```


<div>                            <div id="b247bc65-91aa-46e2-91aa-0ef0b4efec79" class="plotly-graph-div" style="height:600px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("b247bc65-91aa-46e2-91aa-0ef0b4efec79")) {                    Plotly.newPlot(                        "b247bc65-91aa-46e2-91aa-0ef0b4efec79",                        [{"fill":"toself","fillcolor":"rgb(0, 31, 43)","hoverinfo":"name","legendgroup":"rgb(0, 31, 43)","mode":"none","name":"Job 82","showlegend":true,"x":["2022-01-03 00:23:59.160839","2022-01-03 02:29:51.608392","2022-01-03 02:29:51.608392","2022-01-03 00:23:59.160839","2022-01-03 00:23:59.160839","2022-01-11 21:41:11.328671","2022-01-11 23:05:06.293706","2022-01-11 23:05:06.293706","2022-01-11 21:41:11.328671","2022-01-11 21:41:11.328671","2022-01-11 23:11:06.293706","2022-01-11 23:27:53.286713","2022-01-11 23:27:53.286713","2022-01-11 23:11:06.293706","2022-01-11 23:11:06.293706","2022-01-11 23:33:53.286713","2022-01-11 23:59:03.776224","2022-01-11 23:59:03.776224","2022-01-11 23:33:53.286713"],"y":[18.8,18.8,19.2,19.2,null,30.8,30.8,31.2,31.2,null,15.8,15.8,16.2,16.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(1, 19, 61)","hoverinfo":"name","legendgroup":"rgb(1, 19, 61)","mode":"none","name":"Job 32","showlegend":true,"x":["2022-01-03 02:46:38.601399","2022-01-03 03:53:46.573427","2022-01-03 03:53:46.573427","2022-01-03 02:46:38.601399","2022-01-03 02:46:38.601399","2022-01-13 22:04:41.118881","2022-01-14 00:10:33.566434","2022-01-14 00:10:33.566434","2022-01-13 22:04:41.118881","2022-01-13 22:04:41.118881","2022-01-14 05:33:02.937063","2022-01-14 06:40:10.909091","2022-01-14 06:40:10.909091","2022-01-14 05:33:02.937063"],"y":[26.8,26.8,27.2,27.2,null,25.8,25.8,26.2,26.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(100, 36, 174)","hoverinfo":"name","legendgroup":"rgb(100, 36, 174)","mode":"none","name":"Job 120","showlegend":true,"x":["2022-01-02 09:48:52.027972","2022-01-14 08:31:39.860140","2022-01-14 08:31:39.860140","2022-01-02 09:48:52.027972","2022-01-02 09:48:52.027972","2022-02-03 02:59:36.503497","2022-02-04 20:57:05.454545","2022-02-04 20:57:05.454545","2022-02-03 02:59:36.503497"],"y":[7.8,7.8,8.2,8.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(103, 188, 246)","hoverinfo":"name","legendgroup":"rgb(103, 188, 246)","mode":"none","name":"Job 35","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 15:57:07.972028","2022-01-01 15:57:07.972028","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-11 01:16:00.839161","2022-01-11 02:39:55.804196","2022-01-11 02:39:55.804196","2022-01-11 01:16:00.839161","2022-01-11 01:16:00.839161","2022-02-03 02:17:39.020979","2022-02-03 02:59:36.503497","2022-02-03 02:59:36.503497","2022-02-03 02:17:39.020979","2022-02-03 02:17:39.020979","2022-02-03 03:05:36.503497","2022-02-03 03:47:33.986014","2022-02-03 03:47:33.986014","2022-02-03 03:05:36.503497"],"y":[22.8,22.8,23.2,23.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2,null,2.8,2.8,3.2,3.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(104, 213, 69)","hoverinfo":"name","legendgroup":"rgb(104, 213, 69)","mode":"none","name":"Job 78","showlegend":true,"x":["2022-01-04 07:35:18.881119","2022-01-05 04:34:03.356643","2022-01-05 04:34:03.356643","2022-01-04 07:35:18.881119","2022-01-04 07:35:18.881119","2022-01-05 04:40:03.356643","2022-01-05 07:27:53.286713","2022-01-05 07:27:53.286713","2022-01-05 04:40:03.356643","2022-01-05 04:40:03.356643","2022-01-05 07:33:53.286713","2022-01-06 08:44:22.657343","2022-01-06 08:44:22.657343","2022-01-05 07:33:53.286713","2022-01-05 07:33:53.286713","2022-01-11 20:48:26.853147","2022-01-11 22:12:21.818182","2022-01-11 22:12:21.818182","2022-01-11 20:48:26.853147"],"y":[27.8,27.8,28.2,28.2,null,3.8,3.8,4.2,4.2,null,4.8,4.8,5.2,5.2,null,15.8,15.8,16.2,16.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(106, 59, 98)","hoverinfo":"name","legendgroup":"rgb(106, 59, 98)","mode":"none","name":"Job 34","showlegend":true,"x":["2022-01-03 07:23:33.986014","2022-01-03 09:04:15.944056","2022-01-03 09:04:15.944056","2022-01-03 07:23:33.986014","2022-01-03 07:23:33.986014","2022-01-11 13:51:15.524476","2022-01-11 15:57:07.972028","2022-01-11 15:57:07.972028","2022-01-11 13:51:15.524476","2022-01-11 13:51:15.524476","2022-01-19 05:16:15.944056","2022-01-19 06:23:23.916084","2022-01-19 06:23:23.916084","2022-01-19 05:16:15.944056"],"y":[22.8,22.8,23.2,23.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(107, 118, 217)","hoverinfo":"name","legendgroup":"rgb(107, 118, 217)","mode":"none","name":"Job 38","showlegend":true,"x":["2022-01-07 16:42:26.853147","2022-01-08 09:29:26.433566","2022-01-08 09:29:26.433566","2022-01-07 16:42:26.853147","2022-01-07 16:42:26.853147","2022-01-08 09:35:26.433566","2022-01-08 12:23:16.363636","2022-01-08 12:23:16.363636","2022-01-08 09:35:26.433566","2022-01-08 09:35:26.433566","2022-01-08 12:29:16.363636","2022-01-09 10:09:58.321678","2022-01-09 10:09:58.321678","2022-01-08 12:29:16.363636","2022-01-08 12:29:16.363636","2022-01-12 04:13:12.167832","2022-01-12 05:37:07.132867","2022-01-12 05:37:07.132867","2022-01-12 04:13:12.167832"],"y":[30.8,30.8,31.2,31.2,null,3.8,3.8,4.2,4.2,null,-0.2,-0.2,0.2,0.2,null,15.8,15.8,16.2,16.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(11, 153, 214)","hoverinfo":"name","legendgroup":"rgb(11, 153, 214)","mode":"none","name":"Job 63","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 21:49:34.825175","2022-01-01 21:49:34.825175","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-09 14:15:43.216783","2022-01-09 16:29:59.160839","2022-01-09 16:29:59.160839","2022-01-09 14:15:43.216783","2022-01-09 14:15:43.216783","2022-01-12 06:50:15.104895","2022-01-12 20:49:24.755245","2022-01-12 20:49:24.755245","2022-01-12 06:50:15.104895","2022-01-12 06:50:15.104895","2022-01-13 09:16:15.944056","2022-01-13 12:04:05.874126","2022-01-13 12:04:05.874126","2022-01-13 09:16:15.944056"],"y":[7.8,7.8,8.2,8.2,null,0.8,0.8,1.2,1.2,null,5.8,5.8,6.2,6.2,null,28.8,28.8,29.2,29.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(114, 35, 134)","hoverinfo":"name","legendgroup":"rgb(114, 35, 134)","mode":"none","name":"Job 71","showlegend":true,"x":["2022-01-01 21:32:47.832168","2022-01-02 00:20:37.762238","2022-01-02 00:20:37.762238","2022-01-01 21:32:47.832168","2022-01-01 21:32:47.832168","2022-01-02 20:37:24.755245","2022-01-02 22:01:19.720280","2022-01-02 22:01:19.720280","2022-01-02 20:37:24.755245","2022-01-02 20:37:24.755245","2022-02-05 19:36:31.888112","2022-02-05 20:18:29.370629","2022-02-05 20:18:29.370629","2022-02-05 19:36:31.888112"],"y":[22.8,22.8,23.2,23.2,null,27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(116, 11, 97)","hoverinfo":"name","legendgroup":"rgb(116, 11, 97)","mode":"none","name":"Job 5","showlegend":true,"x":["2022-01-11 17:46:13.426573","2022-01-11 18:53:21.398601","2022-01-11 18:53:21.398601","2022-01-11 17:46:13.426573","2022-01-11 17:46:13.426573","2022-01-11 18:59:21.398601","2022-01-11 19:58:05.874126","2022-01-11 19:58:05.874126","2022-01-11 18:59:21.398601","2022-01-11 18:59:21.398601","2022-01-19 04:34:18.461538","2022-01-19 05:16:15.944056","2022-01-19 05:16:15.944056","2022-01-19 04:34:18.461538","2022-01-19 04:34:18.461538","2022-01-19 05:22:15.944056","2022-01-19 06:04:13.426573","2022-01-19 06:04:13.426573","2022-01-19 05:22:15.944056"],"y":[30.8,30.8,31.2,31.2,null,24.8,24.8,25.2,25.2,null,20.8,20.8,21.2,21.2,null,2.8,2.8,3.2,3.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(116, 89, 192)","hoverinfo":"name","legendgroup":"rgb(116, 89, 192)","mode":"none","name":"Job 19","showlegend":true,"x":["2022-01-02 06:29:51.608392","2022-01-02 09:59:39.020979","2022-01-02 09:59:39.020979","2022-01-02 06:29:51.608392","2022-01-02 06:29:51.608392","2022-01-08 10:28:10.909091","2022-01-08 11:35:18.881119","2022-01-08 11:35:18.881119","2022-01-08 10:28:10.909091","2022-01-08 10:28:10.909091","2022-01-11 07:03:40.699301","2022-01-11 08:52:46.153846","2022-01-11 08:52:46.153846","2022-01-11 07:03:40.699301"],"y":[26.8,26.8,27.2,27.2,null,25.8,25.8,26.2,26.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(12, 159, 73)","hoverinfo":"name","legendgroup":"rgb(12, 159, 73)","mode":"none","name":"Job 101","showlegend":true,"x":["2022-01-05 03:52:05.874126","2022-01-06 21:49:34.825175","2022-01-06 21:49:34.825175","2022-01-05 03:52:05.874126","2022-01-05 03:52:05.874126","2022-01-06 21:55:34.825175","2022-01-08 01:53:54.125874","2022-01-08 01:53:54.125874","2022-01-06 21:55:34.825175","2022-01-06 21:55:34.825175","2022-01-10 21:33:02.937063","2022-01-13 16:41:01.258741","2022-01-13 16:41:01.258741","2022-01-10 21:33:02.937063","2022-01-10 21:33:02.937063","2022-01-18 11:05:21.398601","2022-01-19 01:04:31.048951","2022-01-19 01:04:31.048951","2022-01-18 11:05:21.398601"],"y":[25.8,25.8,26.2,26.2,null,8.8,8.8,9.2,9.2,null,9.8,9.8,10.2,10.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(13, 19, 155)","hoverinfo":"name","legendgroup":"rgb(13, 19, 155)","mode":"none","name":"Job 44","showlegend":true,"x":["2022-01-09 12:45:48.251748","2022-01-09 13:52:56.223776","2022-01-09 13:52:56.223776","2022-01-09 12:45:48.251748","2022-01-09 12:45:48.251748","2022-01-09 13:58:56.223776","2022-01-09 14:15:43.216783","2022-01-09 14:15:43.216783","2022-01-09 13:58:56.223776","2022-01-09 13:58:56.223776","2022-01-13 16:21:50.769231","2022-01-13 17:45:45.734266","2022-01-13 17:45:45.734266","2022-01-13 16:21:50.769231"],"y":[30.8,30.8,31.2,31.2,null,0.8,0.8,1.2,1.2,null,-0.2,-0.2,0.2,0.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(132, 86, 253)","hoverinfo":"name","legendgroup":"rgb(132, 86, 253)","mode":"none","name":"Job 102","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-08 14:39:55.804196","2022-01-08 14:39:55.804196","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-08 14:45:55.804196","2022-01-10 08:43:24.755245","2022-01-10 08:43:24.755245","2022-01-08 14:45:55.804196","2022-01-08 14:45:55.804196","2022-01-10 08:49:24.755245","2022-02-07 08:09:07.972028","2022-02-07 08:09:07.972028","2022-01-10 08:49:24.755245","2022-01-10 08:49:24.755245","2022-02-07 08:15:07.972028","2022-02-09 16:11:46.573427","2022-02-09 16:11:46.573427","2022-02-07 08:15:07.972028","2022-02-07 08:15:07.972028","2022-02-09 16:17:46.573427","2022-02-10 13:16:31.048951","2022-02-10 13:16:31.048951","2022-02-09 16:17:46.573427","2022-02-09 16:17:46.573427","2022-02-10 13:22:31.048951","2022-02-11 17:20:50.349650","2022-02-11 17:20:50.349650","2022-02-10 13:22:31.048951"],"y":[28.8,28.8,29.2,29.2,null,2.8,2.8,3.2,3.2,null,1.8,1.8,2.2,2.2,null,13.8,13.8,14.2,14.2,null,15.8,15.8,16.2,16.2,null,12.8,12.8,13.2,13.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(134, 4, 66)","hoverinfo":"name","legendgroup":"rgb(134, 4, 66)","mode":"none","name":"Job 16","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 19:01:44.895105","2022-01-01 19:01:44.895105","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-01 23:02:42.797203","2022-01-02 00:26:37.762238","2022-01-02 00:26:37.762238","2022-01-01 23:02:42.797203","2022-01-01 23:02:42.797203","2022-01-13 11:22:08.391608","2022-01-13 12:46:03.356643","2022-01-13 12:46:03.356643","2022-01-13 11:22:08.391608","2022-01-13 11:22:08.391608","2022-01-13 23:57:23.076923","2022-01-14 02:03:15.524476","2022-01-14 02:03:15.524476","2022-01-13 23:57:23.076923"],"y":[30.8,30.8,31.2,31.2,null,15.8,15.8,16.2,16.2,null,2.8,2.8,3.2,3.2,null,9.8,9.8,10.2,10.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(135, 229, 34)","hoverinfo":"name","legendgroup":"rgb(135, 229, 34)","mode":"none","name":"Job 9","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 15:57:07.972028","2022-01-01 15:57:07.972028","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-02 09:00:54.545455","2022-01-02 09:42:52.027972","2022-01-02 09:42:52.027972","2022-01-02 09:00:54.545455","2022-01-02 09:00:54.545455","2022-01-18 09:16:15.944056","2022-01-18 10:23:23.916084","2022-01-18 10:23:23.916084","2022-01-18 09:16:15.944056","2022-01-18 09:16:15.944056","2022-01-18 11:44:55.384615","2022-01-18 15:14:42.797203","2022-01-18 15:14:42.797203","2022-01-18 11:44:55.384615"],"y":[24.8,24.8,25.2,25.2,null,27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2,null,31.8,31.8,32.2,32.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(136, 135, 160)","hoverinfo":"name","legendgroup":"rgb(136, 135, 160)","mode":"none","name":"Job 28","showlegend":true,"x":["2022-01-01 19:01:44.895105","2022-01-01 23:30:16.783217","2022-01-01 23:30:16.783217","2022-01-01 19:01:44.895105","2022-01-01 19:01:44.895105","2022-01-07 19:47:03.776224","2022-01-07 20:54:11.748252","2022-01-07 20:54:11.748252","2022-01-07 19:47:03.776224","2022-01-07 19:47:03.776224","2022-01-09 11:27:53.286713","2022-01-09 12:09:50.769231","2022-01-09 12:09:50.769231","2022-01-09 11:27:53.286713","2022-01-09 11:27:53.286713","2022-01-14 15:03:40.699301","2022-01-14 15:45:38.181818","2022-01-14 15:45:38.181818","2022-01-14 15:03:40.699301"],"y":[26.8,26.8,27.2,27.2,null,25.8,25.8,26.2,26.2,null,20.8,20.8,21.2,21.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(137, 176, 240)","hoverinfo":"name","legendgroup":"rgb(137, 176, 240)","mode":"none","name":"Job 45","showlegend":true,"x":["2022-01-01 21:49:34.825175","2022-01-01 23:13:29.790210","2022-01-01 23:13:29.790210","2022-01-01 21:49:34.825175","2022-01-01 21:49:34.825175","2022-01-02 19:55:27.272727","2022-01-02 21:02:35.244755","2022-01-02 21:02:35.244755","2022-01-02 19:55:27.272727","2022-01-02 19:55:27.272727","2022-01-13 07:12:46.993007","2022-01-13 07:54:44.475524","2022-01-13 07:54:44.475524","2022-01-13 07:12:46.993007","2022-01-13 07:12:46.993007","2022-01-14 06:40:10.909091","2022-01-14 07:22:08.391608","2022-01-14 07:22:08.391608","2022-01-14 06:40:10.909091"],"y":[30.8,30.8,31.2,31.2,null,24.8,24.8,25.2,25.2,null,15.8,15.8,16.2,16.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(14, 242, 197)","hoverinfo":"name","legendgroup":"rgb(14, 242, 197)","mode":"none","name":"Job 43","showlegend":true,"x":["2022-01-05 01:27:02.937063","2022-01-05 02:50:57.902098","2022-01-05 02:50:57.902098","2022-01-05 01:27:02.937063","2022-01-05 01:27:02.937063","2022-01-09 13:52:56.223776","2022-01-09 14:34:53.706294","2022-01-09 14:34:53.706294","2022-01-09 13:52:56.223776","2022-01-09 13:52:56.223776","2022-01-13 12:46:03.356643","2022-01-13 13:11:13.846154","2022-01-13 13:11:13.846154","2022-01-13 12:46:03.356643","2022-01-13 12:46:03.356643","2022-01-09 14:40:53.706294","2022-01-09 16:04:48.671329","2022-01-09 16:04:48.671329","2022-01-09 14:40:53.706294"],"y":[31.8,31.8,32.2,32.2,null,30.8,30.8,31.2,31.2,null,2.8,2.8,3.2,3.2,null,31.8,31.8,32.2,32.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(142, 4, 253)","hoverinfo":"name","legendgroup":"rgb(142, 4, 253)","mode":"none","name":"Job 46","showlegend":true,"x":["2022-01-01 23:13:29.790210","2022-01-02 04:49:09.650350","2022-01-02 04:49:09.650350","2022-01-01 23:13:29.790210","2022-01-01 23:13:29.790210","2022-01-08 01:34:43.636364","2022-01-08 08:34:18.461538","2022-01-08 08:34:18.461538","2022-01-08 01:34:43.636364","2022-01-08 01:34:43.636364","2022-01-13 05:48:52.027972","2022-01-13 07:12:46.993007","2022-01-13 07:12:46.993007","2022-01-13 05:48:52.027972","2022-01-13 05:48:52.027972","2022-01-19 18:56:15.104895","2022-01-19 21:02:07.552448","2022-01-19 21:02:07.552448","2022-01-19 18:56:15.104895"],"y":[16.8,16.8,17.2,17.2,null,14.8,14.8,15.2,15.2,null,15.8,15.8,16.2,16.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(147, 30, 215)","hoverinfo":"name","legendgroup":"rgb(147, 30, 215)","mode":"none","name":"Job 86","showlegend":true,"x":["2022-01-02 01:36:09.230769","2022-01-02 08:18:57.062937","2022-01-02 08:18:57.062937","2022-01-02 01:36:09.230769","2022-01-02 01:36:09.230769","2022-01-02 08:24:57.062937","2022-01-02 09:48:52.027972","2022-01-02 09:48:52.027972","2022-01-02 08:24:57.062937","2022-01-02 08:24:57.062937","2022-01-09 12:09:50.769231","2022-01-09 14:15:43.216783","2022-01-09 14:15:43.216783","2022-01-09 12:09:50.769231"],"y":[17.8,17.8,18.2,18.2,null,7.8,7.8,8.2,8.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(15, 34, 188)","hoverinfo":"name","legendgroup":"rgb(15, 34, 188)","mode":"none","name":"Job 24","showlegend":true,"x":["2022-01-08 21:47:54.125874","2022-01-09 01:59:39.020979","2022-01-09 01:59:39.020979","2022-01-08 21:47:54.125874","2022-01-08 21:47:54.125874","2022-01-09 10:03:58.321678","2022-01-09 11:27:53.286713","2022-01-09 11:27:53.286713","2022-01-09 10:03:58.321678","2022-01-09 10:03:58.321678","2022-01-13 15:59:03.776224","2022-01-13 17:22:58.741259","2022-01-13 17:22:58.741259","2022-01-13 15:59:03.776224","2022-01-13 15:59:03.776224","2022-01-13 21:23:56.643357","2022-01-14 02:59:36.503497","2022-01-14 02:59:36.503497","2022-01-13 21:23:56.643357"],"y":[30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2,null,2.8,2.8,3.2,3.2,null,5.8,5.8,6.2,6.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(151, 102, 182)","hoverinfo":"name","legendgroup":"rgb(151, 102, 182)","mode":"none","name":"Job 15","showlegend":true,"x":["2022-01-01 20:25:39.860140","2022-01-01 21:49:34.825175","2022-01-01 21:49:34.825175","2022-01-01 20:25:39.860140","2022-01-01 20:25:39.860140","2022-01-12 19:30:16.783217","2022-01-12 22:01:19.720280","2022-01-12 22:01:19.720280","2022-01-12 19:30:16.783217","2022-01-12 19:30:16.783217","2022-01-12 22:07:19.720280","2022-01-12 22:49:17.202797","2022-01-12 22:49:17.202797","2022-01-12 22:07:19.720280","2022-01-12 22:07:19.720280","2022-01-12 22:55:17.202797","2022-01-13 00:35:59.160839","2022-01-13 00:35:59.160839","2022-01-12 22:55:17.202797","2022-01-12 22:55:17.202797","2022-01-13 21:09:33.146853","2022-01-13 23:57:23.076923","2022-01-13 23:57:23.076923","2022-01-13 21:09:33.146853"],"y":[17.8,17.8,18.2,18.2,null,25.8,25.8,26.2,26.2,null,15.8,15.8,16.2,16.2,null,12.8,12.8,13.2,13.2,null,9.8,9.8,10.2,10.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(157, 128, 184)","hoverinfo":"name","legendgroup":"rgb(157, 128, 184)","mode":"none","name":"Job 10","showlegend":true,"x":["2022-01-04 06:28:10.909091","2022-01-04 07:35:18.881119","2022-01-04 07:35:18.881119","2022-01-04 06:28:10.909091","2022-01-04 06:28:10.909091","2022-01-19 06:23:23.916084","2022-01-19 07:05:21.398601","2022-01-19 07:05:21.398601","2022-01-19 06:23:23.916084","2022-01-19 06:23:23.916084","2022-01-19 07:11:21.398601","2022-01-19 08:18:29.370629","2022-01-19 08:18:29.370629","2022-01-19 07:11:21.398601"],"y":[27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2,null,16.8,16.8,17.2,17.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(157, 145, 15)","hoverinfo":"name","legendgroup":"rgb(157, 145, 15)","mode":"none","name":"Job 23","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 19:01:44.895105","2022-01-01 19:01:44.895105","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-06 07:56:25.174825","2022-01-06 09:37:07.132867","2022-01-06 09:37:07.132867","2022-01-06 07:56:25.174825","2022-01-06 07:56:25.174825","2022-01-06 09:43:07.132867","2022-01-06 20:54:26.853147","2022-01-06 20:54:26.853147","2022-01-06 09:43:07.132867","2022-01-06 09:43:07.132867","2022-01-09 02:51:13.006993","2022-01-09 05:39:02.937063","2022-01-09 05:39:02.937063","2022-01-09 02:51:13.006993"],"y":[14.8,14.8,15.2,15.2,null,6.8,6.8,7.2,7.2,null,10.8,10.8,11.2,11.2,null,28.8,28.8,29.2,29.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(162, 107, 85)","hoverinfo":"name","legendgroup":"rgb(162, 107, 85)","mode":"none","name":"Job 31","showlegend":true,"x":["2022-01-04 11:21:53.286713","2022-01-04 13:02:35.244755","2022-01-04 13:02:35.244755","2022-01-04 11:21:53.286713","2022-01-04 11:21:53.286713","2022-01-14 00:18:57.062937","2022-01-14 01:42:52.027972","2022-01-14 01:42:52.027972","2022-01-14 00:18:57.062937","2022-01-14 00:18:57.062937","2022-02-05 20:18:29.370629","2022-02-05 21:00:26.853147","2022-02-05 21:00:26.853147","2022-02-05 20:18:29.370629"],"y":[19.8,19.8,20.2,20.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(162, 2, 232)","hoverinfo":"name","legendgroup":"rgb(162, 2, 232)","mode":"none","name":"Job 114","showlegend":true,"x":["2022-01-02 01:44:32.727273","2022-01-03 05:42:52.027972","2022-01-03 05:42:52.027972","2022-01-02 01:44:32.727273","2022-01-02 01:44:32.727273","2022-01-11 23:05:06.293706","2022-01-13 05:51:15.524476","2022-01-13 05:51:15.524476","2022-01-11 23:05:06.293706","2022-01-11 23:05:06.293706","2022-02-05 06:19:19.720280","2022-02-05 17:30:39.440559","2022-02-05 17:30:39.440559","2022-02-05 06:19:19.720280"],"y":[22.8,22.8,23.2,23.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(164, 79, 82)","hoverinfo":"name","legendgroup":"rgb(164, 79, 82)","mode":"none","name":"Job 36","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 19:01:44.895105","2022-01-01 19:01:44.895105","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-01 19:52:05.874126","2022-01-01 21:16:00.839161","2022-01-01 21:16:00.839161","2022-01-01 19:52:05.874126","2022-01-01 19:52:05.874126","2022-01-06 21:49:34.825175","2022-01-06 23:13:29.790210","2022-01-06 23:13:29.790210","2022-01-06 21:49:34.825175","2022-01-06 21:49:34.825175","2022-01-11 18:00:36.923077","2022-01-11 19:07:44.895105","2022-01-11 19:07:44.895105","2022-01-11 18:00:36.923077"],"y":[26.8,26.8,27.2,27.2,null,31.8,31.8,32.2,32.2,null,25.8,25.8,26.2,26.2,null,15.8,15.8,16.2,16.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(165, 79, 195)","hoverinfo":"name","legendgroup":"rgb(165, 79, 195)","mode":"none","name":"Job 4","showlegend":true,"x":["2022-01-13 07:15:10.489510","2022-01-13 08:05:31.468531","2022-01-13 08:05:31.468531","2022-01-13 07:15:10.489510","2022-01-13 07:15:10.489510","2022-01-13 08:11:31.468531","2022-01-13 08:28:18.461538","2022-01-13 08:28:18.461538","2022-01-13 08:11:31.468531","2022-01-13 08:11:31.468531","2022-01-13 16:41:01.258741","2022-01-13 17:39:45.734266","2022-01-13 17:39:45.734266","2022-01-13 16:41:01.258741"],"y":[30.8,30.8,31.2,31.2,null,2.8,2.8,3.2,3.2,null,9.8,9.8,10.2,10.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(167, 87, 72)","hoverinfo":"name","legendgroup":"rgb(167, 87, 72)","mode":"none","name":"Job 27","showlegend":true,"x":["2022-01-05 02:44:57.902098","2022-01-05 03:52:05.874126","2022-01-05 03:52:05.874126","2022-01-05 02:44:57.902098","2022-01-05 02:44:57.902098","2022-01-06 09:45:30.629371","2022-01-06 10:10:41.118881","2022-01-06 10:10:41.118881","2022-01-06 09:45:30.629371"],"y":[25.8,25.8,26.2,26.2,null,15.8,15.8,16.2,16.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(170, 187, 162)","hoverinfo":"name","legendgroup":"rgb(170, 187, 162)","mode":"none","name":"Job 70","showlegend":true,"x":["2022-01-11 01:07:37.342657","2022-01-11 06:43:17.202797","2022-01-11 06:43:17.202797","2022-01-11 01:07:37.342657","2022-01-11 01:07:37.342657","2022-01-11 12:33:20.559441","2022-01-11 16:03:07.972028","2022-01-11 16:03:07.972028","2022-01-11 12:33:20.559441","2022-01-11 12:33:20.559441","2022-01-11 16:09:07.972028","2022-01-12 00:32:37.762238","2022-01-12 00:32:37.762238","2022-01-11 16:09:07.972028"],"y":[25.8,25.8,26.2,26.2,null,0.8,0.8,1.2,1.2,null,5.8,5.8,6.2,6.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(176, 205, 241)","hoverinfo":"name","legendgroup":"rgb(176, 205, 241)","mode":"none","name":"Job 39","showlegend":true,"x":["2022-01-02 16:00:29.370629","2022-01-03 06:41:36.503497","2022-01-03 06:41:36.503497","2022-01-02 16:00:29.370629","2022-01-02 16:00:29.370629","2022-01-09 10:45:55.804196","2022-01-09 12:26:37.762238","2022-01-09 12:26:37.762238","2022-01-09 10:45:55.804196","2022-01-09 10:45:55.804196","2022-01-09 12:32:37.762238","2022-01-10 07:50:40.279720","2022-01-10 07:50:40.279720","2022-01-09 12:32:37.762238"],"y":[13.8,13.8,14.2,14.2,null,15.8,15.8,16.2,16.2,null,28.8,28.8,29.2,29.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(176, 213, 94)","hoverinfo":"name","legendgroup":"rgb(176, 213, 94)","mode":"none","name":"Job 81","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 20:25:39.860140","2022-01-01 20:25:39.860140","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-06 08:40:46.153846","2022-01-06 10:21:28.111888","2022-01-06 10:21:28.111888","2022-01-06 08:40:46.153846","2022-01-06 08:40:46.153846","2022-01-18 07:27:10.489510","2022-01-18 08:09:07.972028","2022-01-18 08:09:07.972028","2022-01-18 07:27:10.489510"],"y":[17.8,17.8,18.2,18.2,null,27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(178, 127, 166)","hoverinfo":"name","legendgroup":"rgb(178, 127, 166)","mode":"none","name":"Job 72","showlegend":true,"x":["2022-01-02 00:20:37.762238","2022-01-02 01:44:32.727273","2022-01-02 01:44:32.727273","2022-01-02 00:20:37.762238","2022-01-02 00:20:37.762238","2022-01-11 06:43:17.202797","2022-01-11 08:49:09.650350","2022-01-11 08:49:09.650350","2022-01-11 06:43:17.202797","2022-01-11 06:43:17.202797","2022-01-11 08:55:09.650350","2022-01-11 10:02:17.622378","2022-01-11 10:02:17.622378","2022-01-11 08:55:09.650350"],"y":[22.8,22.8,23.2,23.2,null,25.8,25.8,26.2,26.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(179, 168, 226)","hoverinfo":"name","legendgroup":"rgb(179, 168, 226)","mode":"none","name":"Job 7","showlegend":true,"x":["2022-01-01 17:21:02.937063","2022-01-01 18:44:57.902098","2022-01-01 18:44:57.902098","2022-01-01 17:21:02.937063","2022-01-01 17:21:02.937063","2022-01-14 04:22:18.461538","2022-01-14 05:46:13.426573","2022-01-14 05:46:13.426573","2022-01-14 04:22:18.461538"],"y":[31.8,31.8,32.2,32.2,null,25.8,25.8,26.2,26.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(179, 202, 141)","hoverinfo":"name","legendgroup":"rgb(179, 202, 141)","mode":"none","name":"Job 79","showlegend":true,"x":["2022-01-03 06:41:36.503497","2022-01-03 23:28:36.083916","2022-01-03 23:28:36.083916","2022-01-03 06:41:36.503497","2022-01-03 06:41:36.503497","2022-01-12 02:32:30.209790","2022-01-12 04:13:12.167832","2022-01-12 04:13:12.167832","2022-01-12 02:32:30.209790","2022-01-12 02:32:30.209790","2022-01-13 12:04:05.874126","2022-01-14 16:02:25.174825","2022-01-14 16:02:25.174825","2022-01-13 12:04:05.874126"],"y":[13.8,13.8,14.2,14.2,null,15.8,15.8,16.2,16.2,null,28.8,28.8,29.2,29.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(179, 45, 223)","hoverinfo":"name","legendgroup":"rgb(179, 45, 223)","mode":"none","name":"Job 108","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-02 18:48:19.300699","2022-01-02 18:48:19.300699","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-05 09:52:56.223776","2022-01-05 23:52:05.874126","2022-01-05 23:52:05.874126","2022-01-05 09:52:56.223776","2022-01-05 09:52:56.223776","2022-01-09 14:15:43.216783","2022-01-09 21:15:18.041958","2022-01-09 21:15:18.041958","2022-01-09 14:15:43.216783","2022-01-09 14:15:43.216783","2022-01-14 08:04:05.874126","2022-01-14 15:03:40.699301","2022-01-14 15:03:40.699301","2022-01-14 08:04:05.874126"],"y":[18.8,18.8,19.2,19.2,null,27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(18, 187, 67)","hoverinfo":"name","legendgroup":"rgb(18, 187, 67)","mode":"none","name":"Job 11","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 16:13:54.965035","2022-01-01 16:13:54.965035","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-14 00:10:33.566434","2022-01-14 02:16:26.013986","2022-01-14 02:16:26.013986","2022-01-14 00:10:33.566434"],"y":[16.8,16.8,17.2,17.2,null,25.8,25.8,26.2,26.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(18, 248, 215)","hoverinfo":"name","legendgroup":"rgb(18, 248, 215)","mode":"none","name":"Job 107","showlegend":true,"x":["2022-01-08 11:35:18.881119","2022-01-10 23:43:42.377622","2022-01-10 23:43:42.377622","2022-01-08 11:35:18.881119","2022-01-08 11:35:18.881119","2022-01-13 11:07:44.895105","2022-01-13 15:19:29.790210","2022-01-13 15:19:29.790210","2022-01-13 11:07:44.895105"],"y":[25.8,25.8,26.2,26.2,null,15.8,15.8,16.2,16.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(186, 32, 212)","hoverinfo":"name","legendgroup":"rgb(186, 32, 212)","mode":"none","name":"Job 1","showlegend":true,"x":["2022-01-03 09:04:15.944056","2022-01-03 11:35:18.881119","2022-01-03 11:35:18.881119","2022-01-03 09:04:15.944056","2022-01-03 09:04:15.944056","2022-01-13 09:37:49.930070","2022-01-13 11:01:44.895105","2022-01-13 11:01:44.895105","2022-01-13 09:37:49.930070","2022-01-13 09:37:49.930070","2022-02-04 22:21:00.419580","2022-02-04 23:02:57.902098","2022-02-04 23:02:57.902098","2022-02-04 22:21:00.419580"],"y":[22.8,22.8,23.2,23.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(187, 19, 50)","hoverinfo":"name","legendgroup":"rgb(187, 19, 50)","mode":"none","name":"Job 8","showlegend":true,"x":["2022-01-03 07:23:33.986014","2022-01-03 08:05:31.468531","2022-01-03 08:05:31.468531","2022-01-03 07:23:33.986014","2022-01-03 07:23:33.986014","2022-01-13 08:05:31.468531","2022-01-13 08:55:52.447552","2022-01-13 08:55:52.447552","2022-01-13 08:05:31.468531","2022-01-13 08:05:31.468531","2022-02-05 17:30:39.440559","2022-02-05 18:12:36.923077","2022-02-05 18:12:36.923077","2022-02-05 17:30:39.440559","2022-02-05 17:30:39.440559","2022-02-05 18:18:36.923077","2022-02-05 19:00:34.405594","2022-02-05 19:00:34.405594","2022-02-05 18:18:36.923077"],"y":[26.8,26.8,27.2,27.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(189, 21, 77)","hoverinfo":"name","legendgroup":"rgb(189, 21, 77)","mode":"none","name":"Job 91","showlegend":true,"x":["2022-01-03 04:35:44.055944","2022-01-03 07:23:33.986014","2022-01-03 07:23:33.986014","2022-01-03 04:35:44.055944","2022-01-03 04:35:44.055944","2022-01-14 02:16:26.013986","2022-01-14 04:22:18.461538","2022-01-14 04:22:18.461538","2022-01-14 02:16:26.013986"],"y":[17.8,17.8,18.2,18.2,null,25.8,25.8,26.2,26.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(191, 6, 147)","hoverinfo":"name","legendgroup":"rgb(191, 6, 147)","mode":"none","name":"Job 98","showlegend":true,"x":["2022-01-08 04:18:57.062937","2022-01-08 10:36:34.405594","2022-01-08 10:36:34.405594","2022-01-08 04:18:57.062937","2022-01-08 04:18:57.062937","2022-01-09 08:40:03.356643","2022-01-09 10:03:58.321678","2022-01-09 10:03:58.321678","2022-01-09 08:40:03.356643","2022-01-09 08:40:03.356643","2022-01-13 06:11:39.020979","2022-01-13 08:59:28.951049","2022-01-13 08:59:28.951049","2022-01-13 06:11:39.020979","2022-01-13 06:11:39.020979","2022-01-13 09:05:28.951049","2022-01-13 10:12:36.923077","2022-01-13 10:12:36.923077","2022-01-13 09:05:28.951049"],"y":[27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2,null,12.8,12.8,13.2,13.2,null,18.8,18.8,19.2,19.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(191, 80, 121)","hoverinfo":"name","legendgroup":"rgb(191, 80, 121)","mode":"none","name":"Job 60","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 20:25:39.860140","2022-01-01 20:25:39.860140","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-01 20:31:39.860140","2022-01-03 11:41:18.881119","2022-01-03 11:41:18.881119","2022-01-01 20:31:39.860140","2022-01-01 20:31:39.860140","2022-01-03 11:47:18.881119","2022-01-03 18:46:53.706294","2022-01-03 18:46:53.706294","2022-01-03 11:47:18.881119","2022-01-03 11:47:18.881119","2022-01-19 01:46:28.531469","2022-01-19 04:34:18.461538","2022-01-19 04:34:18.461538","2022-01-19 01:46:28.531469","2022-01-19 01:46:28.531469","2022-01-19 04:40:18.461538","2022-01-19 06:46:10.909091","2022-01-19 06:46:10.909091","2022-01-19 04:40:18.461538"],"y":[21.8,21.8,22.2,22.2,null,10.8,10.8,11.2,11.2,null,29.8,29.8,30.2,30.2,null,20.8,20.8,21.2,21.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(194, 20, 126)","hoverinfo":"name","legendgroup":"rgb(194, 20, 126)","mode":"none","name":"Job 84","showlegend":true,"x":["2022-01-07 16:17:16.363636","2022-01-07 18:23:08.811189","2022-01-07 18:23:08.811189","2022-01-07 16:17:16.363636","2022-01-07 16:17:16.363636","2022-01-07 18:29:08.811189","2022-01-07 18:45:55.804196","2022-01-07 18:45:55.804196","2022-01-07 18:29:08.811189","2022-01-07 18:29:08.811189","2022-01-14 02:03:15.524476","2022-01-14 03:10:23.496503","2022-01-14 03:10:23.496503","2022-01-14 02:03:15.524476"],"y":[25.8,25.8,26.2,26.2,null,6.8,6.8,7.2,7.2,null,9.8,9.8,10.2,10.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(195, 250, 163)","hoverinfo":"name","legendgroup":"rgb(195, 250, 163)","mode":"none","name":"Job 80","showlegend":true,"x":["2022-01-02 02:18:06.713287","2022-01-03 00:40:46.153846","2022-01-03 00:40:46.153846","2022-01-02 02:18:06.713287","2022-01-02 02:18:06.713287","2022-01-11 13:57:15.524476","2022-01-11 19:32:55.384615","2022-01-11 19:32:55.384615","2022-01-11 13:57:15.524476"],"y":[29.8,29.8,30.2,30.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(195, 45, 130)","hoverinfo":"name","legendgroup":"rgb(195, 45, 130)","mode":"none","name":"Job 106","showlegend":true,"x":["2022-01-02 22:01:19.720280","2022-01-04 04:47:28.951049","2022-01-04 04:47:28.951049","2022-01-02 22:01:19.720280","2022-01-02 22:01:19.720280","2022-01-04 04:53:28.951049","2022-01-05 17:15:18.041958","2022-01-05 17:15:18.041958","2022-01-04 04:53:28.951049","2022-01-04 04:53:28.951049","2022-01-08 23:34:36.083916","2022-01-09 06:34:10.909091","2022-01-09 06:34:10.909091","2022-01-08 23:34:36.083916","2022-01-08 23:34:36.083916","2022-01-09 12:07:27.272727","2022-01-09 19:07:02.097902","2022-01-09 19:07:02.097902","2022-01-09 12:07:27.272727","2022-01-09 12:07:27.272727","2022-01-14 10:57:55.804196","2022-01-15 12:08:25.174825","2022-01-15 12:08:25.174825","2022-01-14 10:57:55.804196"],"y":[27.8,27.8,28.2,28.2,null,9.8,9.8,10.2,10.2,null,20.8,20.8,21.2,21.2,null,3.8,3.8,4.2,4.2,null,9.8,9.8,10.2,10.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(196, 210, 6)","hoverinfo":"name","legendgroup":"rgb(196, 210, 6)","mode":"none","name":"Job 67","showlegend":true,"x":["2022-01-11 08:15:35.664336","2022-01-11 13:51:15.524476","2022-01-11 13:51:15.524476","2022-01-11 08:15:35.664336","2022-01-11 08:15:35.664336","2022-01-18 08:51:05.454545","2022-01-18 09:16:15.944056","2022-01-18 09:16:15.944056","2022-01-18 08:51:05.454545"],"y":[30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(197, 17, 140)","hoverinfo":"name","legendgroup":"rgb(197, 17, 140)","mode":"none","name":"Job 116","showlegend":true,"x":["2022-01-01 16:13:54.965035","2022-01-04 11:21:53.286713","2022-01-04 11:21:53.286713","2022-01-01 16:13:54.965035","2022-01-01 16:13:54.965035","2022-01-04 11:27:53.286713","2022-01-05 01:27:02.937063","2022-01-05 01:27:02.937063","2022-01-04 11:27:53.286713","2022-01-04 11:27:53.286713","2022-01-06 10:21:28.111888","2022-01-07 14:19:47.412587","2022-01-07 14:19:47.412587","2022-01-06 10:21:28.111888","2022-01-06 10:21:28.111888","2022-01-11 02:45:55.804196","2022-01-11 13:57:15.524476","2022-01-11 13:57:15.524476","2022-01-11 02:45:55.804196"],"y":[19.8,19.8,20.2,20.2,null,31.8,31.8,32.2,32.2,null,27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(2, 106, 106)","hoverinfo":"name","legendgroup":"rgb(2, 106, 106)","mode":"none","name":"Job 37","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 16:13:54.965035","2022-01-01 16:13:54.965035","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-02 01:02:35.244755","2022-01-02 01:44:32.727273","2022-01-02 01:44:32.727273","2022-01-02 01:02:35.244755","2022-01-02 01:02:35.244755","2022-01-08 17:08:35.244755","2022-01-08 17:58:56.223776","2022-01-08 17:58:56.223776","2022-01-08 17:08:35.244755","2022-01-08 17:08:35.244755","2022-01-18 15:14:42.797203","2022-01-18 15:56:40.279720","2022-01-18 15:56:40.279720","2022-01-18 15:14:42.797203","2022-01-18 15:14:42.797203","2022-01-08 18:04:56.223776","2022-01-08 18:46:53.706294","2022-01-08 18:46:53.706294","2022-01-08 18:04:56.223776"],"y":[19.8,19.8,20.2,20.2,null,30.8,30.8,31.2,31.2,null,15.8,15.8,16.2,16.2,null,31.8,31.8,32.2,32.2,null,19.8,19.8,20.2,20.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(2, 227, 92)","hoverinfo":"name","legendgroup":"rgb(2, 227, 92)","mode":"none","name":"Job 113","showlegend":true,"x":["2022-01-02 04:49:09.650350","2022-01-05 09:44:32.727273","2022-01-05 09:44:32.727273","2022-01-02 04:49:09.650350","2022-01-02 04:49:09.650350","2022-01-05 23:52:05.874126","2022-01-05 23:52:05.874126","2022-01-05 23:52:05.874126","2022-01-05 23:52:05.874126","2022-01-05 23:52:05.874126","2022-01-08 20:46:46.153846","2022-01-09 10:45:55.804196","2022-01-09 10:45:55.804196","2022-01-08 20:46:46.153846","2022-01-08 20:46:46.153846","2022-01-09 21:23:41.538462","2022-01-11 12:33:20.559441","2022-01-11 12:33:20.559441","2022-01-09 21:23:41.538462","2022-01-09 21:23:41.538462","2022-01-13 08:05:31.468531","2022-01-13 22:04:41.118881","2022-01-13 22:04:41.118881","2022-01-13 08:05:31.468531","2022-01-13 08:05:31.468531","2022-01-13 22:10:41.118881","2022-01-14 09:22:00.839161","2022-01-14 09:22:00.839161","2022-01-13 22:10:41.118881"],"y":[16.8,16.8,17.2,17.2,null,27.8,27.8,28.2,28.2,null,15.8,15.8,16.2,16.2,null,12.8,12.8,13.2,13.2,null,25.8,25.8,26.2,26.2,null,22.8,22.8,23.2,23.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(201, 134, 35)","hoverinfo":"name","legendgroup":"rgb(201, 134, 35)","mode":"none","name":"Job 41","showlegend":true,"x":["2022-01-01 15:31:57.482517","2022-01-01 18:19:47.412587","2022-01-01 18:19:47.412587","2022-01-01 15:31:57.482517","2022-01-01 15:31:57.482517","2022-01-13 05:51:15.524476","2022-01-13 07:15:10.489510","2022-01-13 07:15:10.489510","2022-01-13 05:51:15.524476","2022-01-13 05:51:15.524476","2022-01-18 08:09:07.972028","2022-01-18 08:51:05.454545","2022-01-18 08:51:05.454545","2022-01-18 08:09:07.972028"],"y":[23.8,23.8,24.2,24.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(205, 226, 10)","hoverinfo":"name","legendgroup":"rgb(205, 226, 10)","mode":"none","name":"Job 65","showlegend":true,"x":["2022-01-02 02:01:19.720280","2022-01-02 09:00:54.545455","2022-01-02 09:00:54.545455","2022-01-02 02:01:19.720280","2022-01-02 02:01:19.720280","2022-01-05 09:58:56.223776","2022-01-05 11:39:38.181818","2022-01-05 11:39:38.181818","2022-01-05 09:58:56.223776","2022-01-05 09:58:56.223776","2022-01-11 18:53:21.398601","2022-01-11 21:41:11.328671","2022-01-11 21:41:11.328671","2022-01-11 18:53:21.398601","2022-01-11 18:53:21.398601","2022-01-11 21:47:11.328671","2022-01-11 23:11:06.293706","2022-01-11 23:11:06.293706","2022-01-11 21:47:11.328671","2022-01-11 21:47:11.328671","2022-01-14 05:30:39.440559","2022-01-14 11:06:19.300699","2022-01-14 11:06:19.300699","2022-01-14 05:30:39.440559"],"y":[27.8,27.8,28.2,28.2,null,15.8,15.8,16.2,16.2,null,30.8,30.8,31.2,31.2,null,6.8,6.8,7.2,7.2,null,5.8,5.8,6.2,6.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(207, 105, 64)","hoverinfo":"name","legendgroup":"rgb(207, 105, 64)","mode":"none","name":"Job 54","showlegend":true,"x":["2022-01-01 18:19:47.412587","2022-01-02 02:43:17.202797","2022-01-02 02:43:17.202797","2022-01-01 18:19:47.412587","2022-01-01 18:19:47.412587","2022-01-02 15:01:44.895105","2022-01-02 20:37:24.755245","2022-01-02 20:37:24.755245","2022-01-02 15:01:44.895105","2022-01-02 15:01:44.895105","2022-01-08 17:58:56.223776","2022-01-08 20:46:46.153846","2022-01-08 20:46:46.153846","2022-01-08 17:58:56.223776","2022-01-08 17:58:56.223776","2022-01-08 20:52:46.153846","2022-01-08 23:57:23.076923","2022-01-08 23:57:23.076923","2022-01-08 20:52:46.153846","2022-01-08 20:52:46.153846","2022-01-09 00:03:23.076923","2022-01-09 02:51:13.006993","2022-01-09 02:51:13.006993","2022-01-09 00:03:23.076923"],"y":[23.8,23.8,24.2,24.2,null,27.8,27.8,28.2,28.2,null,15.8,15.8,16.2,16.2,null,6.8,6.8,7.2,7.2,null,28.8,28.8,29.2,29.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(209, 21, 165)","hoverinfo":"name","legendgroup":"rgb(209, 21, 165)","mode":"none","name":"Job 40","showlegend":true,"x":["2022-01-08 14:45:55.804196","2022-01-09 10:20:45.314685","2022-01-09 10:20:45.314685","2022-01-08 14:45:55.804196","2022-01-08 14:45:55.804196","2022-01-19 13:22:58.741259","2022-01-19 16:52:46.153846","2022-01-19 16:52:46.153846","2022-01-19 13:22:58.741259"],"y":[21.8,21.8,22.2,22.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(210, 75, 89)","hoverinfo":"name","legendgroup":"rgb(210, 75, 89)","mode":"none","name":"Job 33","showlegend":true,"x":["2022-01-01 16:13:54.965035","2022-01-01 23:13:29.790210","2022-01-01 23:13:29.790210","2022-01-01 16:13:54.965035","2022-01-01 16:13:54.965035","2022-01-08 12:17:16.363636","2022-01-08 21:47:54.125874","2022-01-08 21:47:54.125874","2022-01-08 12:17:16.363636","2022-01-08 12:17:16.363636","2022-01-08 22:10:41.118881","2022-01-08 23:34:36.083916","2022-01-08 23:34:36.083916","2022-01-08 22:10:41.118881","2022-01-08 22:10:41.118881","2022-01-13 00:35:59.160839","2022-01-13 04:05:46.573427","2022-01-13 04:05:46.573427","2022-01-13 00:35:59.160839","2022-01-13 00:35:59.160839","2022-01-13 18:07:19.720280","2022-01-13 19:14:27.692308","2022-01-13 19:14:27.692308","2022-01-13 18:07:19.720280","2022-01-13 18:07:19.720280","2022-01-13 16:37:24.755245","2022-01-13 18:01:19.720280","2022-01-13 18:01:19.720280","2022-01-13 16:37:24.755245"],"y":[16.8,16.8,17.2,17.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2,null,12.8,12.8,13.2,13.2,null,17.8,17.8,18.2,18.2,null,30.8,30.8,31.2,31.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(215, 81, 4)","hoverinfo":"name","legendgroup":"rgb(215, 81, 4)","mode":"none","name":"Job 104","showlegend":true,"x":["2022-01-04 18:46:38.601399","2022-01-07 16:42:26.853147","2022-01-07 16:42:26.853147","2022-01-04 18:46:38.601399","2022-01-04 18:46:38.601399","2022-01-07 16:48:26.853147","2022-01-08 20:46:46.153846","2022-01-08 20:46:46.153846","2022-01-07 16:48:26.853147","2022-01-07 16:48:26.853147","2022-01-08 20:52:46.153846","2022-01-09 10:51:55.804196","2022-01-09 10:51:55.804196","2022-01-08 20:52:46.153846","2022-01-08 20:52:46.153846","2022-01-09 10:57:55.804196","2022-01-11 21:42:24.335664","2022-01-11 21:42:24.335664","2022-01-09 10:57:55.804196"],"y":[30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2,null,0.8,0.8,1.2,1.2,null,10.8,10.8,11.2,11.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(216, 241, 158)","hoverinfo":"name","legendgroup":"rgb(216, 241, 158)","mode":"none","name":"Job 111","showlegend":true,"x":["2022-01-02 04:57:33.146853","2022-01-03 13:07:37.342657","2022-01-03 13:07:37.342657","2022-01-02 04:57:33.146853","2022-01-02 04:57:33.146853","2022-01-06 23:13:29.790210","2022-01-07 13:12:39.440559","2022-01-07 13:12:39.440559","2022-01-06 23:13:29.790210","2022-01-06 23:13:29.790210","2022-01-12 22:49:17.202797","2022-01-13 05:48:52.027972","2022-01-13 05:48:52.027972","2022-01-12 22:49:17.202797"],"y":[23.8,23.8,24.2,24.2,null,25.8,25.8,26.2,26.2,null,15.8,15.8,16.2,16.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(22, 148, 1)","hoverinfo":"name","legendgroup":"rgb(22, 148, 1)","mode":"none","name":"Job 47","showlegend":true,"x":["2022-01-09 21:15:18.041958","2022-01-09 22:56:00","2022-01-09 22:56:00","2022-01-09 21:15:18.041958","2022-01-09 21:15:18.041958","2022-01-13 11:01:44.895105","2022-01-13 12:25:39.860140","2022-01-13 12:25:39.860140","2022-01-13 11:01:44.895105"],"y":[31.8,31.8,32.2,32.2,null,30.8,30.8,31.2,31.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(22, 189, 194)","hoverinfo":"name","legendgroup":"rgb(22, 189, 194)","mode":"none","name":"Job 68","showlegend":true,"x":["2022-01-01 23:55:27.272727","2022-01-02 01:36:09.230769","2022-01-02 01:36:09.230769","2022-01-01 23:55:27.272727","2022-01-01 23:55:27.272727","2022-01-07 18:23:08.811189","2022-01-07 19:47:03.776224","2022-01-07 19:47:03.776224","2022-01-07 18:23:08.811189","2022-01-07 18:23:08.811189","2022-01-18 10:23:23.916084","2022-01-18 11:05:21.398601","2022-01-18 11:05:21.398601","2022-01-18 10:23:23.916084","2022-01-18 10:23:23.916084","2022-01-18 11:11:21.398601","2022-01-18 11:53:18.881119","2022-01-18 11:53:18.881119","2022-01-18 11:11:21.398601"],"y":[17.8,17.8,18.2,18.2,null,25.8,25.8,26.2,26.2,null,20.8,20.8,21.2,21.2,null,12.8,12.8,13.2,13.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(22, 37, 0)","hoverinfo":"name","legendgroup":"rgb(22, 37, 0)","mode":"none","name":"Job 110","showlegend":true,"x":["2022-01-02 01:44:32.727273","2022-01-04 16:40:46.153846","2022-01-04 16:40:46.153846","2022-01-02 01:44:32.727273","2022-01-02 01:44:32.727273","2022-01-04 16:46:46.153846","2022-01-06 07:56:25.174825","2022-01-06 07:56:25.174825","2022-01-04 16:46:46.153846","2022-01-04 16:46:46.153846","2022-01-06 08:02:25.174825","2022-01-10 09:56:32.727273","2022-01-10 09:56:32.727273","2022-01-06 08:02:25.174825"],"y":[30.8,30.8,31.2,31.2,null,6.8,6.8,7.2,7.2,null,11.8,11.8,12.2,12.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(220, 119, 34)","hoverinfo":"name","legendgroup":"rgb(220, 119, 34)","mode":"none","name":"Job 87","showlegend":true,"x":["2022-01-09 18:10:41.118881","2022-01-09 21:15:18.041958","2022-01-09 21:15:18.041958","2022-01-09 18:10:41.118881","2022-01-09 18:10:41.118881","2022-01-13 21:47:54.125874","2022-01-13 23:11:49.090909","2022-01-13 23:11:49.090909","2022-01-13 21:47:54.125874"],"y":[31.8,31.8,32.2,32.2,null,30.8,30.8,31.2,31.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(220, 38, 126)","hoverinfo":"name","legendgroup":"rgb(220, 38, 126)","mode":"none","name":"Job 12","showlegend":true,"x":["2022-01-02 08:18:57.062937","2022-01-02 13:12:39.440559","2022-01-02 13:12:39.440559","2022-01-02 08:18:57.062937","2022-01-02 08:18:57.062937","2022-01-10 15:34:36.083916","2022-01-10 16:41:44.055944","2022-01-10 16:41:44.055944","2022-01-10 15:34:36.083916"],"y":[8.8,8.8,9.2,9.2,null,15.8,15.8,16.2,16.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(224, 24, 147)","hoverinfo":"name","legendgroup":"rgb(224, 24, 147)","mode":"none","name":"Job 6","showlegend":true,"x":["2022-01-02 14:11:23.916084","2022-01-02 18:23:08.811189","2022-01-02 18:23:08.811189","2022-01-02 14:11:23.916084","2022-01-02 14:11:23.916084","2022-01-08 01:28:43.636364","2022-01-08 06:22:26.013986","2022-01-08 06:22:26.013986","2022-01-08 01:28:43.636364","2022-01-08 01:28:43.636364","2022-01-12 05:37:07.132867","2022-01-12 06:44:15.104895","2022-01-12 06:44:15.104895","2022-01-12 05:37:07.132867","2022-01-12 05:37:07.132867","2022-01-12 06:50:15.104895","2022-01-12 08:30:57.062937","2022-01-12 08:30:57.062937","2022-01-12 06:50:15.104895"],"y":[26.8,26.8,27.2,27.2,null,21.8,21.8,22.2,22.2,null,15.8,15.8,16.2,16.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(225, 120, 131)","hoverinfo":"name","legendgroup":"rgb(225, 120, 131)","mode":"none","name":"Job 61","showlegend":true,"x":["2022-01-09 06:11:23.916084","2022-01-09 09:32:47.832168","2022-01-09 09:32:47.832168","2022-01-09 06:11:23.916084","2022-01-09 06:11:23.916084","2022-01-09 09:38:47.832168","2022-01-09 11:44:40.279720","2022-01-09 11:44:40.279720","2022-01-09 09:38:47.832168","2022-01-09 09:38:47.832168","2022-01-13 13:00:26.853147","2022-01-13 21:23:56.643357","2022-01-13 21:23:56.643357","2022-01-13 13:00:26.853147","2022-01-13 13:00:26.853147","2022-02-05 18:12:36.923077","2022-02-05 19:36:31.888112","2022-02-05 19:36:31.888112","2022-02-05 18:12:36.923077"],"y":[30.8,30.8,31.2,31.2,null,29.8,29.8,30.2,30.2,null,5.8,5.8,6.2,6.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(225, 224, 138)","hoverinfo":"name","legendgroup":"rgb(225, 224, 138)","mode":"none","name":"Job 13","showlegend":true,"x":["2022-01-01 19:01:44.895105","2022-01-01 23:13:29.790210","2022-01-01 23:13:29.790210","2022-01-01 19:01:44.895105","2022-01-01 19:01:44.895105","2022-01-09 06:34:10.909091","2022-01-09 07:58:05.874126","2022-01-09 07:58:05.874126","2022-01-09 06:34:10.909091","2022-01-09 06:34:10.909091","2022-01-09 08:04:05.874126","2022-01-09 09:53:11.328671","2022-01-09 09:53:11.328671","2022-01-09 08:04:05.874126"],"y":[14.8,14.8,15.2,15.2,null,20.8,20.8,21.2,21.2,null,4.8,4.8,5.2,5.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(225, 62, 236)","hoverinfo":"name","legendgroup":"rgb(225, 62, 236)","mode":"none","name":"Job 90","showlegend":true,"x":["2022-01-10 11:33:38.181818","2022-01-10 14:04:41.118881","2022-01-10 14:04:41.118881","2022-01-10 11:33:38.181818","2022-01-10 11:33:38.181818","2022-01-10 14:10:41.118881","2022-01-10 14:52:38.601399","2022-01-10 14:52:38.601399","2022-01-10 14:10:41.118881","2022-01-10 14:10:41.118881","2022-01-10 14:58:38.601399","2022-01-10 16:05:46.573427","2022-01-10 16:05:46.573427","2022-01-10 14:58:38.601399"],"y":[30.8,30.8,31.2,31.2,null,15.8,15.8,16.2,16.2,null,17.8,17.8,18.2,18.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(227, 138, 198)","hoverinfo":"name","legendgroup":"rgb(227, 138, 198)","mode":"none","name":"Job 55","showlegend":true,"x":["2022-01-02 01:36:09.230769","2022-01-02 03:42:01.678322","2022-01-02 03:42:01.678322","2022-01-02 01:36:09.230769","2022-01-02 01:36:09.230769","2022-01-05 23:52:05.874126","2022-01-06 02:39:55.804196","2022-01-06 02:39:55.804196","2022-01-05 23:52:05.874126","2022-01-05 23:52:05.874126","2022-01-10 14:52:38.601399","2022-01-10 15:34:36.083916","2022-01-10 15:34:36.083916","2022-01-10 14:52:38.601399","2022-01-10 14:52:38.601399","2022-01-14 03:52:20.979021","2022-01-14 05:33:02.937063","2022-01-14 05:33:02.937063","2022-01-14 03:52:20.979021","2022-01-14 03:52:20.979021","2022-01-14 05:39:02.937063","2022-01-14 08:26:52.867133","2022-01-14 08:26:52.867133","2022-01-14 05:39:02.937063"],"y":[26.8,26.8,27.2,27.2,null,27.8,27.8,28.2,28.2,null,15.8,15.8,16.2,16.2,null,0.8,0.8,1.2,1.2,null,-0.2,-0.2,0.2,0.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(227, 235, 11)","hoverinfo":"name","legendgroup":"rgb(227, 235, 11)","mode":"none","name":"Job 99","showlegend":true,"x":["2022-01-02 18:48:19.300699","2022-01-03 00:23:59.160839","2022-01-03 00:23:59.160839","2022-01-02 18:48:19.300699","2022-01-02 18:48:19.300699","2022-01-13 12:25:39.860140","2022-01-13 15:13:29.790210","2022-01-13 15:13:29.790210","2022-01-13 12:25:39.860140","2022-01-13 12:25:39.860140","2022-01-19 21:02:07.552448","2022-01-19 22:51:13.006993","2022-01-19 22:51:13.006993","2022-01-19 21:02:07.552448"],"y":[18.8,18.8,19.2,19.2,null,30.8,30.8,31.2,31.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(228, 240, 124)","hoverinfo":"name","legendgroup":"rgb(228, 240, 124)","mode":"none","name":"Job 20","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 19:18:31.888112","2022-01-01 19:18:31.888112","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-12 05:23:56.643357","2022-01-13 13:34:00.839161","2022-01-13 13:34:00.839161","2022-01-12 05:23:56.643357","2022-01-12 05:23:56.643357","2022-01-13 13:40:00.839161","2022-01-13 19:15:40.699301","2022-01-13 19:15:40.699301","2022-01-13 13:40:00.839161","2022-01-13 13:40:00.839161","2022-02-05 04:13:27.272727","2022-02-05 06:19:19.720280","2022-02-05 06:19:19.720280","2022-02-05 04:13:27.272727","2022-02-05 04:13:27.272727","2022-02-05 06:25:19.720280","2022-02-05 08:31:12.167832","2022-02-05 08:31:12.167832","2022-02-05 06:25:19.720280"],"y":[29.8,29.8,30.2,30.2,null,-0.2,-0.2,0.2,0.2,null,21.8,21.8,22.2,22.2,null,20.8,20.8,21.2,21.2,null,2.8,2.8,3.2,3.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(231, 216, 25)","hoverinfo":"name","legendgroup":"rgb(231, 216, 25)","mode":"none","name":"Job 57","showlegend":true,"x":["2022-01-08 01:05:56.643357","2022-01-08 04:10:33.566434","2022-01-08 04:10:33.566434","2022-01-08 01:05:56.643357","2022-01-08 01:05:56.643357","2022-01-08 12:48:26.853147","2022-01-08 13:55:34.825175","2022-01-08 13:55:34.825175","2022-01-08 12:48:26.853147","2022-01-08 12:48:26.853147","2022-01-11 19:41:18.881119","2022-01-11 21:05:13.846154","2022-01-11 21:05:13.846154","2022-01-11 19:41:18.881119","2022-01-11 19:41:18.881119","2022-01-11 21:11:13.846154","2022-01-11 22:51:55.804196","2022-01-11 22:51:55.804196","2022-01-11 21:11:13.846154"],"y":[25.8,25.8,26.2,26.2,null,15.8,15.8,16.2,16.2,null,2.8,2.8,3.2,3.2,null,23.8,23.8,24.2,24.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(234, 202, 35)","hoverinfo":"name","legendgroup":"rgb(234, 202, 35)","mode":"none","name":"Job 100","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 23:13:29.790210","2022-01-01 23:13:29.790210","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-01 23:19:29.790210","2022-01-04 21:15:18.041958","2022-01-04 21:15:18.041958","2022-01-01 23:19:29.790210","2022-01-01 23:19:29.790210","2022-01-04 21:21:18.041958","2022-01-05 03:38:55.384615","2022-01-05 03:38:55.384615","2022-01-04 21:21:18.041958","2022-01-04 21:21:18.041958","2022-01-08 14:37:32.307692","2022-01-08 17:08:35.244755","2022-01-08 17:08:35.244755","2022-01-08 14:37:32.307692","2022-01-08 14:37:32.307692","2022-01-13 04:05:46.573427","2022-01-13 06:11:39.020979","2022-01-13 06:11:39.020979","2022-01-13 04:05:46.573427"],"y":[8.8,8.8,9.2,9.2,null,4.8,4.8,5.2,5.2,null,14.8,14.8,15.2,15.2,null,15.8,15.8,16.2,16.2,null,12.8,12.8,13.2,13.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(235, 253, 30)","hoverinfo":"name","legendgroup":"rgb(235, 253, 30)","mode":"none","name":"Job 74","showlegend":true,"x":["2022-01-01 23:30:16.783217","2022-01-02 01:36:09.230769","2022-01-02 01:36:09.230769","2022-01-01 23:30:16.783217","2022-01-01 23:30:16.783217","2022-01-08 09:29:26.433566","2022-01-08 12:17:16.363636","2022-01-08 12:17:16.363636","2022-01-08 09:29:26.433566","2022-01-08 09:29:26.433566","2022-01-13 15:19:29.790210","2022-01-13 16:26:37.762238","2022-01-13 16:26:37.762238","2022-01-13 15:19:29.790210"],"y":[26.8,26.8,27.2,27.2,null,30.8,30.8,31.2,31.2,null,15.8,15.8,16.2,16.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(235, 37, 200)","hoverinfo":"name","legendgroup":"rgb(235, 37, 200)","mode":"none","name":"Job 59","showlegend":true,"x":["2022-01-02 09:59:39.020979","2022-01-02 14:11:23.916084","2022-01-02 14:11:23.916084","2022-01-02 09:59:39.020979","2022-01-02 09:59:39.020979","2022-01-10 23:43:42.377622","2022-01-11 01:07:37.342657","2022-01-11 01:07:37.342657","2022-01-10 23:43:42.377622","2022-01-10 23:43:42.377622","2022-01-11 17:01:52.447552","2022-01-11 18:50:57.902098","2022-01-11 18:50:57.902098","2022-01-11 17:01:52.447552"],"y":[26.8,26.8,27.2,27.2,null,25.8,25.8,26.2,26.2,null,12.8,12.8,13.2,13.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(242, 153, 176)","hoverinfo":"name","legendgroup":"rgb(242, 153, 176)","mode":"none","name":"Job 112","showlegend":true,"x":["2022-01-02 21:02:35.244755","2022-01-03 11:01:44.895105","2022-01-03 11:01:44.895105","2022-01-02 21:02:35.244755","2022-01-02 21:02:35.244755","2022-01-09 14:34:53.706294","2022-01-10 11:33:38.181818","2022-01-10 11:33:38.181818","2022-01-09 14:34:53.706294","2022-01-09 14:34:53.706294","2022-01-11 23:59:03.776224","2022-01-12 11:10:23.496503","2022-01-12 11:10:23.496503","2022-01-11 23:59:03.776224"],"y":[24.8,24.8,25.2,25.2,null,30.8,30.8,31.2,31.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(243, 197, 144)","hoverinfo":"name","legendgroup":"rgb(243, 197, 144)","mode":"none","name":"Job 42","showlegend":true,"x":["2022-01-02 08:18:57.062937","2022-01-02 09:00:54.545455","2022-01-02 09:00:54.545455","2022-01-02 08:18:57.062937","2022-01-02 08:18:57.062937","2022-01-12 22:01:19.720280","2022-01-12 23:25:14.685315","2022-01-12 23:25:14.685315","2022-01-12 22:01:19.720280","2022-01-12 22:01:19.720280","2022-01-13 08:45:05.454545","2022-01-13 09:01:52.447552","2022-01-13 09:01:52.447552","2022-01-13 08:45:05.454545","2022-01-13 08:45:05.454545","2022-01-19 06:04:13.426573","2022-01-19 06:29:23.916084","2022-01-19 06:29:23.916084","2022-01-19 06:04:13.426573"],"y":[17.8,17.8,18.2,18.2,null,25.8,25.8,26.2,26.2,null,15.8,15.8,16.2,16.2,null,2.8,2.8,3.2,3.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(245, 29, 232)","hoverinfo":"name","legendgroup":"rgb(245, 29, 232)","mode":"none","name":"Job 103","showlegend":true,"x":["2022-01-05 03:38:55.384615","2022-01-08 01:34:43.636364","2022-01-08 01:34:43.636364","2022-01-05 03:38:55.384615","2022-01-05 03:38:55.384615","2022-01-10 08:43:24.755245","2022-01-11 19:41:18.881119","2022-01-11 19:41:18.881119","2022-01-10 08:43:24.755245","2022-01-10 08:43:24.755245","2022-01-11 19:47:18.881119","2022-01-20 13:34:43.636364","2022-01-20 13:34:43.636364","2022-01-11 19:47:18.881119","2022-01-11 19:47:18.881119","2022-01-20 13:40:43.636364","2022-01-21 17:39:02.937063","2022-01-21 17:39:02.937063","2022-01-20 13:40:43.636364"],"y":[14.8,14.8,15.2,15.2,null,2.8,2.8,3.2,3.2,null,4.8,4.8,5.2,5.2,null,29.8,29.8,30.2,30.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(246, 74, 41)","hoverinfo":"name","legendgroup":"rgb(246, 74, 41)","mode":"none","name":"Job 83","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 17:21:02.937063","2022-01-01 17:21:02.937063","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-06 07:58:48.671329","2022-01-06 08:40:46.153846","2022-01-06 08:40:46.153846","2022-01-06 07:58:48.671329","2022-01-06 07:58:48.671329","2022-01-12 11:10:23.496503","2022-01-12 11:35:33.986014","2022-01-12 11:35:33.986014","2022-01-12 11:10:23.496503","2022-01-12 11:10:23.496503","2022-01-09 16:04:48.671329","2022-01-09 18:10:41.118881","2022-01-09 18:10:41.118881","2022-01-09 16:04:48.671329"],"y":[31.8,31.8,32.2,32.2,null,27.8,27.8,28.2,28.2,null,6.8,6.8,7.2,7.2,null,31.8,31.8,32.2,32.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(247, 251, 13)","hoverinfo":"name","legendgroup":"rgb(247, 251, 13)","mode":"none","name":"Job 115","showlegend":true,"x":["2022-01-03 07:23:33.986014","2022-01-04 04:22:18.461538","2022-01-04 04:22:18.461538","2022-01-03 07:23:33.986014","2022-01-03 07:23:33.986014","2022-01-13 01:05:56.643357","2022-01-13 08:05:31.468531","2022-01-13 08:05:31.468531","2022-01-13 01:05:56.643357","2022-01-13 01:05:56.643357","2022-01-13 08:11:31.468531","2022-01-13 16:35:01.258741","2022-01-13 16:35:01.258741","2022-01-13 08:11:31.468531","2022-01-13 08:11:31.468531","2022-01-13 16:41:01.258741","2022-01-14 03:52:20.979021","2022-01-14 03:52:20.979021","2022-01-13 16:41:01.258741"],"y":[17.8,17.8,18.2,18.2,null,25.8,25.8,26.2,26.2,null,20.8,20.8,21.2,21.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(249, 156, 65)","hoverinfo":"name","legendgroup":"rgb(249, 156, 65)","mode":"none","name":"Job 109","showlegend":true,"x":["2022-01-08 10:36:34.405594","2022-01-09 07:35:18.881119","2022-01-09 07:35:18.881119","2022-01-08 10:36:34.405594","2022-01-08 10:36:34.405594","2022-01-09 07:41:18.881119","2022-01-09 18:52:38.601399","2022-01-09 18:52:38.601399","2022-01-09 07:41:18.881119","2022-01-09 07:41:18.881119","2022-01-09 18:58:38.601399","2022-01-10 21:33:02.937063","2022-01-10 21:33:02.937063","2022-01-09 18:58:38.601399","2022-01-09 18:58:38.601399","2022-01-12 22:24:06.713287","2022-01-13 08:11:31.468531","2022-01-13 08:11:31.468531","2022-01-12 22:24:06.713287"],"y":[27.8,27.8,28.2,28.2,null,12.8,12.8,13.2,13.2,null,9.8,9.8,10.2,10.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(252, 125, 21)","hoverinfo":"name","legendgroup":"rgb(252, 125, 21)","mode":"none","name":"Job 96","showlegend":true,"x":["2022-01-09 01:59:39.020979","2022-01-09 06:11:23.916084","2022-01-09 06:11:23.916084","2022-01-09 01:59:39.020979","2022-01-09 01:59:39.020979","2022-01-12 21:16:58.741259","2022-01-12 22:24:06.713287","2022-01-12 22:24:06.713287","2022-01-12 21:16:58.741259","2022-01-12 21:16:58.741259","2022-01-13 13:11:13.846154","2022-01-13 15:59:03.776224","2022-01-13 15:59:03.776224","2022-01-13 13:11:13.846154","2022-01-13 13:11:13.846154","2022-01-13 17:39:45.734266","2022-01-13 21:09:33.146853","2022-01-13 21:09:33.146853","2022-01-13 17:39:45.734266"],"y":[30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2,null,2.8,2.8,3.2,3.2,null,9.8,9.8,10.2,10.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(252, 142, 168)","hoverinfo":"name","legendgroup":"rgb(252, 142, 168)","mode":"none","name":"Job 30","showlegend":true,"x":["2022-01-14 01:42:52.027972","2022-01-14 06:11:23.916084","2022-01-14 06:11:23.916084","2022-01-14 01:42:52.027972","2022-01-14 01:42:52.027972","2022-01-19 15:09:40.699301","2022-01-19 17:57:30.629371","2022-01-19 17:57:30.629371","2022-01-19 15:09:40.699301","2022-01-19 15:09:40.699301","2022-01-19 18:03:30.629371","2022-01-20 00:21:07.972028","2022-01-20 00:21:07.972028","2022-01-19 18:03:30.629371"],"y":[30.8,30.8,31.2,31.2,null,6.8,6.8,7.2,7.2,null,-0.2,-0.2,0.2,0.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(254, 212, 212)","hoverinfo":"name","legendgroup":"rgb(254, 212, 212)","mode":"none","name":"Job 53","showlegend":true,"x":["2022-01-08 08:34:18.461538","2022-01-08 14:09:58.321678","2022-01-08 14:09:58.321678","2022-01-08 08:34:18.461538","2022-01-08 08:34:18.461538","2022-01-13 09:43:49.930070","2022-01-13 11:07:44.895105","2022-01-13 11:07:44.895105","2022-01-13 09:43:49.930070","2022-01-13 09:43:49.930070","2022-01-14 03:10:23.496503","2022-01-14 05:58:13.426573","2022-01-14 05:58:13.426573","2022-01-14 03:10:23.496503"],"y":[14.8,14.8,15.2,15.2,null,15.8,15.8,16.2,16.2,null,9.8,9.8,10.2,10.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(255, 1, 148)","hoverinfo":"name","legendgroup":"rgb(255, 1, 148)","mode":"none","name":"Job 97","showlegend":true,"x":["2022-01-02 09:42:52.027972","2022-01-02 13:54:36.923077","2022-01-02 13:54:36.923077","2022-01-02 09:42:52.027972","2022-01-02 09:42:52.027972","2022-01-09 12:26:37.762238","2022-01-09 13:33:45.734266","2022-01-09 13:33:45.734266","2022-01-09 12:26:37.762238","2022-01-09 12:26:37.762238","2022-01-11 11:09:25.594406","2022-01-11 12:33:20.559441","2022-01-11 12:33:20.559441","2022-01-11 11:09:25.594406","2022-01-11 11:09:25.594406","2022-01-11 12:39:20.559441","2022-01-11 14:20:02.517483","2022-01-11 14:20:02.517483","2022-01-11 12:39:20.559441"],"y":[27.8,27.8,28.2,28.2,null,15.8,15.8,16.2,16.2,null,0.8,0.8,1.2,1.2,null,16.8,16.8,17.2,17.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(29, 131, 212)","hoverinfo":"name","legendgroup":"rgb(29, 131, 212)","mode":"none","name":"Job 75","showlegend":true,"x":["2022-01-03 05:17:41.538462","2022-01-03 06:41:36.503497","2022-01-03 06:41:36.503497","2022-01-03 05:17:41.538462","2022-01-03 05:17:41.538462","2022-01-12 23:25:14.685315","2022-01-13 01:05:56.643357","2022-01-13 01:05:56.643357","2022-01-12 23:25:14.685315","2022-01-12 23:25:14.685315","2022-01-13 07:54:44.475524","2022-01-13 08:45:05.454545","2022-01-13 08:45:05.454545","2022-01-13 07:54:44.475524","2022-01-13 07:54:44.475524","2022-01-14 16:27:35.664336","2022-01-14 17:34:43.636364","2022-01-14 17:34:43.636364","2022-01-14 16:27:35.664336"],"y":[26.8,26.8,27.2,27.2,null,25.8,25.8,26.2,26.2,null,15.8,15.8,16.2,16.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(30, 32, 230)","hoverinfo":"name","legendgroup":"rgb(30, 32, 230)","mode":"none","name":"Job 105","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-05 02:44:57.902098","2022-01-05 02:44:57.902098","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-05 02:50:57.902098","2022-01-05 23:49:42.377622","2022-01-05 23:49:42.377622","2022-01-05 02:50:57.902098","2022-01-05 02:50:57.902098","2022-01-12 12:53:28.951049","2022-01-13 02:52:38.601399","2022-01-13 02:52:38.601399","2022-01-12 12:53:28.951049","2022-01-12 12:53:28.951049","2022-01-13 02:58:38.601399","2022-01-16 00:54:26.853147","2022-01-16 00:54:26.853147","2022-01-13 02:58:38.601399","2022-01-13 02:58:38.601399","2022-01-11 08:49:09.650350","2022-01-12 12:47:28.951049","2022-01-12 12:47:28.951049","2022-01-11 08:49:09.650350"],"y":[25.8,25.8,26.2,26.2,null,20.8,20.8,21.2,21.2,null,6.8,6.8,7.2,7.2,null,10.8,10.8,11.2,11.2,null,25.8,25.8,26.2,26.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(32, 137, 126)","hoverinfo":"name","legendgroup":"rgb(32, 137, 126)","mode":"none","name":"Job 69","showlegend":true,"x":["2022-01-08 09:04:15.944056","2022-01-08 10:28:10.909091","2022-01-08 10:28:10.909091","2022-01-08 09:04:15.944056","2022-01-08 09:04:15.944056","2022-01-11 10:02:17.622378","2022-01-11 11:09:25.594406","2022-01-11 11:09:25.594406","2022-01-11 10:02:17.622378","2022-01-11 10:02:17.622378","2022-01-12 03:20:27.692308","2022-01-12 06:50:15.104895","2022-01-12 06:50:15.104895","2022-01-12 03:20:27.692308","2022-01-12 03:20:27.692308","2022-01-19 07:05:21.398601","2022-01-19 08:04:05.874126","2022-01-19 08:04:05.874126","2022-01-19 07:05:21.398601"],"y":[25.8,25.8,26.2,26.2,null,0.8,0.8,1.2,1.2,null,5.8,5.8,6.2,6.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(35, 212, 8)","hoverinfo":"name","legendgroup":"rgb(35, 212, 8)","mode":"none","name":"Job 56","showlegend":true,"x":["2022-01-01 20:25:39.860140","2022-01-02 02:01:19.720280","2022-01-02 02:01:19.720280","2022-01-01 20:25:39.860140","2022-01-01 20:25:39.860140","2022-01-08 20:46:46.153846","2022-01-08 22:10:41.118881","2022-01-08 22:10:41.118881","2022-01-08 20:46:46.153846","2022-01-08 20:46:46.153846","2022-01-11 15:21:10.489510","2022-01-11 17:01:52.447552","2022-01-11 17:01:52.447552","2022-01-11 15:21:10.489510","2022-01-11 15:21:10.489510","2022-01-12 00:32:37.762238","2022-01-12 03:20:27.692308","2022-01-12 03:20:27.692308","2022-01-12 00:32:37.762238"],"y":[27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2,null,12.8,12.8,13.2,13.2,null,5.8,5.8,6.2,6.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(35, 26, 177)","hoverinfo":"name","legendgroup":"rgb(35, 26, 177)","mode":"none","name":"Job 62","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-02 04:49:09.650350","2022-01-02 04:49:09.650350","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-02 04:55:09.650350","2022-01-02 09:06:54.545455","2022-01-02 09:06:54.545455","2022-01-02 04:55:09.650350","2022-01-02 04:55:09.650350","2022-01-02 09:12:54.545455","2022-01-04 10:09:58.321678","2022-01-04 10:09:58.321678","2022-01-02 09:12:54.545455","2022-01-02 09:12:54.545455","2022-01-08 06:22:26.013986","2022-01-08 14:45:55.804196","2022-01-08 14:45:55.804196","2022-01-08 06:22:26.013986","2022-01-08 06:22:26.013986","2022-01-19 08:04:05.874126","2022-01-19 10:09:58.321678","2022-01-19 10:09:58.321678","2022-01-19 08:04:05.874126","2022-01-19 08:04:05.874126","2022-01-19 10:15:58.321678","2022-01-19 13:03:48.251748","2022-01-19 13:03:48.251748","2022-01-19 10:15:58.321678"],"y":[13.8,13.8,14.2,14.2,null,12.8,12.8,13.2,13.2,null,11.8,11.8,12.2,12.2,null,21.8,21.8,22.2,22.2,null,20.8,20.8,21.2,21.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(4, 208, 74)","hoverinfo":"name","legendgroup":"rgb(4, 208, 74)","mode":"none","name":"Job 94","showlegend":true,"x":["2022-01-02 17:24:24.335664","2022-01-03 04:35:44.055944","2022-01-03 04:35:44.055944","2022-01-02 17:24:24.335664","2022-01-02 17:24:24.335664","2022-01-12 12:47:28.951049","2022-01-12 18:06:21.818182","2022-01-12 18:06:21.818182","2022-01-12 12:47:28.951049","2022-01-12 12:47:28.951049","2022-01-12 18:12:21.818182","2022-01-12 21:16:58.741259","2022-01-12 21:16:58.741259","2022-01-12 18:12:21.818182","2022-01-12 18:12:21.818182","2022-01-13 02:52:38.601399","2022-01-13 06:22:26.013986","2022-01-13 06:22:26.013986","2022-01-13 02:52:38.601399","2022-01-13 02:52:38.601399","2022-01-13 06:28:26.013986","2022-01-13 09:16:15.944056","2022-01-13 09:16:15.944056","2022-01-13 06:28:26.013986"],"y":[17.8,17.8,18.2,18.2,null,25.8,25.8,26.2,26.2,null,20.8,20.8,21.2,21.2,null,6.8,6.8,7.2,7.2,null,28.8,28.8,29.2,29.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(41, 39, 121)","hoverinfo":"name","legendgroup":"rgb(41, 39, 121)","mode":"none","name":"Job 77","showlegend":true,"x":["2022-01-03 05:42:52.027972","2022-01-03 07:23:33.986014","2022-01-03 07:23:33.986014","2022-01-03 05:42:52.027972","2022-01-03 05:42:52.027972","2022-01-11 15:57:07.972028","2022-01-11 17:04:15.944056","2022-01-11 17:04:15.944056","2022-01-11 15:57:07.972028","2022-01-11 15:57:07.972028","2022-01-11 17:10:15.944056","2022-01-11 18:00:36.923077","2022-01-11 18:00:36.923077","2022-01-11 17:10:15.944056","2022-01-11 17:10:15.944056","2022-01-11 18:06:36.923077","2022-01-11 19:13:44.895105","2022-01-11 19:13:44.895105","2022-01-11 18:06:36.923077","2022-01-11 18:06:36.923077","2022-01-19 06:04:13.426573","2022-01-19 06:46:10.909091","2022-01-19 06:46:10.909091","2022-01-19 06:04:13.426573"],"y":[22.8,22.8,23.2,23.2,null,30.8,30.8,31.2,31.2,null,15.8,15.8,16.2,16.2,null,26.8,26.8,27.2,27.2,null,31.8,31.8,32.2,32.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(46, 247, 226)","hoverinfo":"name","legendgroup":"rgb(46, 247, 226)","mode":"none","name":"Job 14","showlegend":true,"x":["2022-01-02 18:23:08.811189","2022-01-03 01:05:56.643357","2022-01-03 01:05:56.643357","2022-01-02 18:23:08.811189","2022-01-02 18:23:08.811189","2022-01-08 04:10:33.566434","2022-01-08 09:04:15.944056","2022-01-08 09:04:15.944056","2022-01-08 04:10:33.566434","2022-01-08 04:10:33.566434","2022-01-09 21:15:18.041958","2022-01-09 22:56:00","2022-01-09 22:56:00","2022-01-09 21:15:18.041958","2022-01-09 21:15:18.041958","2022-01-11 12:33:20.559441","2022-01-11 15:21:10.489510","2022-01-11 15:21:10.489510","2022-01-11 12:33:20.559441","2022-01-11 12:33:20.559441","2022-01-11 15:27:10.489510","2022-01-11 17:58:13.426573","2022-01-11 17:58:13.426573","2022-01-11 15:27:10.489510"],"y":[26.8,26.8,27.2,27.2,null,25.8,25.8,26.2,26.2,null,20.8,20.8,21.2,21.2,null,12.8,12.8,13.2,13.2,null,14.8,14.8,15.2,15.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(47, 43, 189)","hoverinfo":"name","legendgroup":"rgb(47, 43, 189)","mode":"none","name":"Job 66","showlegend":true,"x":["2022-01-04 16:40:46.153846","2022-01-04 18:46:38.601399","2022-01-04 18:46:38.601399","2022-01-04 16:40:46.153846","2022-01-04 16:40:46.153846","2022-01-04 18:52:38.601399","2022-01-04 21:40:28.531469","2022-01-04 21:40:28.531469","2022-01-04 18:52:38.601399","2022-01-04 18:52:38.601399","2022-01-13 17:16:58.741259","2022-01-13 17:58:56.223776","2022-01-13 17:58:56.223776","2022-01-13 17:16:58.741259","2022-01-13 17:16:58.741259","2022-01-14 07:22:08.391608","2022-01-14 08:04:05.874126","2022-01-14 08:04:05.874126","2022-01-14 07:22:08.391608","2022-01-14 07:22:08.391608","2022-01-14 08:10:05.874126","2022-01-14 10:57:55.804196","2022-01-14 10:57:55.804196","2022-01-14 08:10:05.874126"],"y":[30.8,30.8,31.2,31.2,null,5.8,5.8,6.2,6.2,null,20.8,20.8,21.2,21.2,null,0.8,0.8,1.2,1.2,null,9.8,9.8,10.2,10.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(50, 21, 217)","hoverinfo":"name","legendgroup":"rgb(50, 21, 217)","mode":"none","name":"Job 88","showlegend":true,"x":["2022-01-01 21:49:34.825175","2022-01-01 23:55:27.272727","2022-01-01 23:55:27.272727","2022-01-01 21:49:34.825175","2022-01-01 21:49:34.825175","2022-01-12 18:06:21.818182","2022-01-12 19:30:16.783217","2022-01-12 19:30:16.783217","2022-01-12 18:06:21.818182","2022-01-12 18:06:21.818182","2022-02-05 03:31:29.790210","2022-02-05 04:13:27.272727","2022-02-05 04:13:27.272727","2022-02-05 03:31:29.790210","2022-02-05 03:31:29.790210","2022-02-05 04:19:27.272727","2022-02-05 05:01:24.755245","2022-02-05 05:01:24.755245","2022-02-05 04:19:27.272727"],"y":[17.8,17.8,18.2,18.2,null,25.8,25.8,26.2,26.2,null,20.8,20.8,21.2,21.2,null,3.8,3.8,4.2,4.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(51, 45, 236)","hoverinfo":"name","legendgroup":"rgb(51, 45, 236)","mode":"none","name":"Job 92","showlegend":true,"x":["2022-01-01 23:13:29.790210","2022-01-02 08:18:57.062937","2022-01-02 08:18:57.062937","2022-01-01 23:13:29.790210","2022-01-01 23:13:29.790210","2022-02-05 00:01:42.377622","2022-02-05 01:25:37.342657","2022-02-05 01:25:37.342657","2022-02-05 00:01:42.377622"],"y":[8.8,8.8,9.2,9.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(52, 113, 143)","hoverinfo":"name","legendgroup":"rgb(52, 113, 143)","mode":"none","name":"Job 95","showlegend":true,"x":["2022-01-02 03:42:01.678322","2022-01-02 06:29:51.608392","2022-01-02 06:29:51.608392","2022-01-02 03:42:01.678322","2022-01-02 03:42:01.678322","2022-01-07 13:12:39.440559","2022-01-07 16:17:16.363636","2022-01-07 16:17:16.363636","2022-01-07 13:12:39.440559","2022-01-07 13:12:39.440559","2022-01-09 07:58:05.874126","2022-01-09 08:40:03.356643","2022-01-09 08:40:03.356643","2022-01-09 07:58:05.874126","2022-01-09 07:58:05.874126","2022-01-09 08:46:03.356643","2022-01-09 10:26:45.314685","2022-01-09 10:26:45.314685","2022-01-09 08:46:03.356643","2022-01-09 08:46:03.356643","2022-01-13 13:34:00.839161","2022-01-13 16:21:50.769231","2022-01-13 16:21:50.769231","2022-01-13 13:34:00.839161"],"y":[26.8,26.8,27.2,27.2,null,25.8,25.8,26.2,26.2,null,20.8,20.8,21.2,21.2,null,3.8,3.8,4.2,4.2,null,-0.2,-0.2,0.2,0.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(56, 168, 62)","hoverinfo":"name","legendgroup":"rgb(56, 168, 62)","mode":"none","name":"Job 21","showlegend":true,"x":["2022-01-07 20:54:11.748252","2022-01-07 23:42:01.678322","2022-01-07 23:42:01.678322","2022-01-07 20:54:11.748252","2022-01-07 20:54:11.748252","2022-01-07 23:48:01.678322","2022-01-08 01:28:43.636364","2022-01-08 01:28:43.636364","2022-01-07 23:48:01.678322","2022-01-07 23:48:01.678322","2022-01-09 10:09:58.321678","2022-01-09 15:45:38.181818","2022-01-09 15:45:38.181818","2022-01-09 10:09:58.321678","2022-01-09 10:09:58.321678","2022-02-04 20:57:05.454545","2022-02-04 22:21:00.419580","2022-02-04 22:21:00.419580","2022-02-04 20:57:05.454545"],"y":[25.8,25.8,26.2,26.2,null,21.8,21.8,22.2,22.2,null,-0.2,-0.2,0.2,0.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(57, 187, 49)","hoverinfo":"name","legendgroup":"rgb(57, 187, 49)","mode":"none","name":"Job 50","showlegend":true,"x":["2022-01-07 23:42:01.678322","2022-01-08 01:05:56.643357","2022-01-08 01:05:56.643357","2022-01-07 23:42:01.678322","2022-01-07 23:42:01.678322","2022-01-08 13:55:34.825175","2022-01-08 14:37:32.307692","2022-01-08 14:37:32.307692","2022-01-08 13:55:34.825175","2022-01-08 13:55:34.825175","2022-01-08 14:43:32.307692","2022-01-08 15:50:40.279720","2022-01-08 15:50:40.279720","2022-01-08 14:43:32.307692"],"y":[25.8,25.8,26.2,26.2,null,15.8,15.8,16.2,16.2,null,23.8,23.8,24.2,24.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(57, 228, 159)","hoverinfo":"name","legendgroup":"rgb(57, 228, 159)","mode":"none","name":"Job 118","showlegend":true,"x":["2022-01-09 07:35:18.881119","2022-01-18 15:21:53.286713","2022-01-18 15:21:53.286713","2022-01-09 07:35:18.881119","2022-01-09 07:35:18.881119","2022-01-18 15:27:53.286713","2022-01-21 13:23:41.538462","2022-01-21 13:23:41.538462","2022-01-18 15:27:53.286713","2022-01-18 15:27:53.286713","2022-01-21 13:29:41.538462","2022-02-02 12:12:29.370629","2022-02-02 12:12:29.370629","2022-01-21 13:29:41.538462","2022-01-21 13:29:41.538462","2022-02-02 12:18:29.370629","2022-02-03 02:17:39.020979","2022-02-03 02:17:39.020979","2022-02-02 12:18:29.370629"],"y":[27.8,27.8,28.2,28.2,null,3.8,3.8,4.2,4.2,null,11.8,11.8,12.2,12.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(6, 92, 92)","hoverinfo":"name","legendgroup":"rgb(6, 92, 92)","mode":"none","name":"Job 2","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 15:31:57.482517","2022-01-01 15:31:57.482517","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-02 13:54:36.923077","2022-01-02 15:01:44.895105","2022-01-02 15:01:44.895105","2022-01-02 13:54:36.923077","2022-01-02 13:54:36.923077","2022-01-05 11:39:38.181818","2022-01-05 11:56:25.174825","2022-01-05 11:56:25.174825","2022-01-05 11:39:38.181818","2022-01-05 11:39:38.181818","2022-01-19 17:57:30.629371","2022-01-19 18:14:17.622378","2022-01-19 18:14:17.622378","2022-01-19 17:57:30.629371"],"y":[23.8,23.8,24.2,24.2,null,27.8,27.8,28.2,28.2,null,15.8,15.8,16.2,16.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(61, 240, 53)","hoverinfo":"name","legendgroup":"rgb(61, 240, 53)","mode":"none","name":"Job 29","showlegend":true,"x":["2022-01-09 09:32:47.832168","2022-01-09 10:39:55.804196","2022-01-09 10:39:55.804196","2022-01-09 09:32:47.832168","2022-01-09 09:32:47.832168","2022-01-09 20:16:33.566434","2022-01-09 21:23:41.538462","2022-01-09 21:23:41.538462","2022-01-09 20:16:33.566434","2022-01-09 20:16:33.566434","2022-01-09 21:29:41.538462","2022-01-10 00:34:18.461538","2022-01-10 00:34:18.461538","2022-01-09 21:29:41.538462","2022-01-09 21:29:41.538462","2022-02-04 23:02:57.902098","2022-02-05 00:01:42.377622","2022-02-05 00:01:42.377622","2022-02-04 23:02:57.902098"],"y":[30.8,30.8,31.2,31.2,null,12.8,12.8,13.2,13.2,null,4.8,4.8,5.2,5.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(63, 8, 11)","hoverinfo":"name","legendgroup":"rgb(63, 8, 11)","mode":"none","name":"Job 26","showlegend":true,"x":["2022-01-04 04:47:28.951049","2022-01-04 06:28:10.909091","2022-01-04 06:28:10.909091","2022-01-04 04:47:28.951049","2022-01-04 04:47:28.951049","2022-01-04 06:34:10.909091","2022-01-04 09:05:13.846154","2022-01-04 09:05:13.846154","2022-01-04 06:34:10.909091","2022-01-04 06:34:10.909091","2022-01-13 09:01:52.447552","2022-01-13 09:43:49.930070","2022-01-13 09:43:49.930070","2022-01-13 09:01:52.447552","2022-01-13 09:01:52.447552","2022-01-13 09:49:49.930070","2022-01-13 10:31:47.412587","2022-01-13 10:31:47.412587","2022-01-13 09:49:49.930070","2022-01-13 09:49:49.930070","2022-01-13 10:37:47.412587","2022-01-13 13:00:26.853147","2022-01-13 13:00:26.853147","2022-01-13 10:37:47.412587"],"y":[27.8,27.8,28.2,28.2,null,1.8,1.8,2.2,2.2,null,15.8,15.8,16.2,16.2,null,2.8,2.8,3.2,3.2,null,5.8,5.8,6.2,6.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(65, 217, 136)","hoverinfo":"name","legendgroup":"rgb(65, 217, 136)","mode":"none","name":"Job 22","showlegend":true,"x":["2022-01-02 04:49:09.650350","2022-01-02 16:00:29.370629","2022-01-02 16:00:29.370629","2022-01-02 04:49:09.650350","2022-01-02 04:49:09.650350","2022-01-05 07:27:53.286713","2022-01-05 11:39:38.181818","2022-01-05 11:39:38.181818","2022-01-05 07:27:53.286713","2022-01-05 07:27:53.286713","2022-01-05 11:45:38.181818","2022-01-07 02:55:17.202797","2022-01-07 02:55:17.202797","2022-01-05 11:45:38.181818","2022-01-05 11:45:38.181818","2022-01-08 14:09:58.321678","2022-01-08 19:45:38.181818","2022-01-08 19:45:38.181818","2022-01-08 14:09:58.321678","2022-01-08 14:09:58.321678","2022-01-19 10:09:58.321678","2022-01-19 12:15:50.769231","2022-01-19 12:15:50.769231","2022-01-19 10:09:58.321678","2022-01-19 10:09:58.321678","2022-01-19 12:21:50.769231","2022-01-19 15:09:40.699301","2022-01-19 15:09:40.699301","2022-01-19 12:21:50.769231"],"y":[13.8,13.8,14.2,14.2,null,3.8,3.8,4.2,4.2,null,5.8,5.8,6.2,6.2,null,14.8,14.8,15.2,15.2,null,20.8,20.8,21.2,21.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(66, 158, 132)","hoverinfo":"name","legendgroup":"rgb(66, 158, 132)","mode":"none","name":"Job 52","showlegend":true,"x":["2022-01-08 01:53:54.125874","2022-01-08 08:53:28.951049","2022-01-08 08:53:28.951049","2022-01-08 01:53:54.125874","2022-01-08 01:53:54.125874","2022-02-05 01:25:37.342657","2022-02-05 02:49:32.307692","2022-02-05 02:49:32.307692","2022-02-05 01:25:37.342657"],"y":[8.8,8.8,9.2,9.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(71, 108, 219)","hoverinfo":"name","legendgroup":"rgb(71, 108, 219)","mode":"none","name":"Job 89","showlegend":true,"x":["2022-01-02 02:43:17.202797","2022-01-02 04:57:33.146853","2022-01-02 04:57:33.146853","2022-01-02 02:43:17.202797","2022-01-02 02:43:17.202797","2022-01-05 08:45:48.251748","2022-01-05 09:52:56.223776","2022-01-05 09:52:56.223776","2022-01-05 08:45:48.251748","2022-01-05 08:45:48.251748","2022-01-19 01:04:31.048951","2022-01-19 01:46:28.531469","2022-01-19 01:46:28.531469","2022-01-19 01:04:31.048951","2022-01-19 01:04:31.048951","2022-01-19 01:52:28.531469","2022-01-19 06:04:13.426573","2022-01-19 06:04:13.426573","2022-01-19 01:52:28.531469"],"y":[23.8,23.8,24.2,24.2,null,27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2,null,31.8,31.8,32.2,32.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(71, 247, 110)","hoverinfo":"name","legendgroup":"rgb(71, 247, 110)","mode":"none","name":"Job 18","showlegend":true,"x":["2022-01-05 04:34:03.356643","2022-01-05 08:45:48.251748","2022-01-05 08:45:48.251748","2022-01-05 04:34:03.356643","2022-01-05 04:34:03.356643","2022-01-05 08:51:48.251748","2022-01-05 09:58:56.223776","2022-01-05 09:58:56.223776","2022-01-05 08:51:48.251748","2022-01-05 08:51:48.251748","2022-01-09 10:26:45.314685","2022-01-09 12:07:27.272727","2022-01-09 12:07:27.272727","2022-01-09 10:26:45.314685","2022-01-09 10:26:45.314685","2022-01-09 12:13:27.272727","2022-01-09 13:20:35.244755","2022-01-09 13:20:35.244755","2022-01-09 12:13:27.272727"],"y":[27.8,27.8,28.2,28.2,null,15.8,15.8,16.2,16.2,null,3.8,3.8,4.2,4.2,null,23.8,23.8,24.2,24.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(76, 120, 97)","hoverinfo":"name","legendgroup":"rgb(76, 120, 97)","mode":"none","name":"Job 117","showlegend":true,"x":["2022-01-01 15:57:07.972028","2022-01-02 19:55:27.272727","2022-01-02 19:55:27.272727","2022-01-01 15:57:07.972028","2022-01-01 15:57:07.972028","2022-01-07 14:19:47.412587","2022-01-08 04:18:57.062937","2022-01-08 04:18:57.062937","2022-01-07 14:19:47.412587","2022-01-07 14:19:47.412587","2022-01-08 04:24:57.062937","2022-01-08 12:48:26.853147","2022-01-08 12:48:26.853147","2022-01-08 04:24:57.062937","2022-01-08 04:24:57.062937","2022-01-08 12:54:26.853147","2022-01-09 00:05:46.573427","2022-01-09 00:05:46.573427","2022-01-08 12:54:26.853147","2022-01-08 12:54:26.853147","2022-01-19 06:46:10.909091","2022-01-19 13:45:45.734266","2022-01-19 13:45:45.734266","2022-01-19 06:46:10.909091"],"y":[24.8,24.8,25.2,25.2,null,27.8,27.8,28.2,28.2,null,15.8,15.8,16.2,16.2,null,18.8,18.8,19.2,19.2,null,31.8,31.8,32.2,32.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(8, 35, 51)","hoverinfo":"name","legendgroup":"rgb(8, 35, 51)","mode":"none","name":"Job 64","showlegend":true,"x":["2022-01-11 02:39:55.804196","2022-01-11 08:15:35.664336","2022-01-11 08:15:35.664336","2022-01-11 02:39:55.804196","2022-01-11 02:39:55.804196","2022-01-11 19:07:44.895105","2022-01-11 20:48:26.853147","2022-01-11 20:48:26.853147","2022-01-11 19:07:44.895105","2022-01-11 19:07:44.895105","2022-01-11 20:54:26.853147","2022-01-11 22:18:21.818182","2022-01-11 22:18:21.818182","2022-01-11 20:54:26.853147","2022-01-11 20:54:26.853147","2022-01-11 22:24:21.818182","2022-01-12 05:23:56.643357","2022-01-12 05:23:56.643357","2022-01-11 22:24:21.818182"],"y":[30.8,30.8,31.2,31.2,null,15.8,15.8,16.2,16.2,null,3.8,3.8,4.2,4.2,null,-0.2,-0.2,0.2,0.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(80, 199, 222)","hoverinfo":"name","legendgroup":"rgb(80, 199, 222)","mode":"none","name":"Job 93","showlegend":true,"x":["2022-01-01 19:18:31.888112","2022-01-02 02:18:06.713287","2022-01-02 02:18:06.713287","2022-01-01 19:18:31.888112","2022-01-01 19:18:31.888112","2022-01-11 23:27:53.286713","2022-01-12 02:32:30.209790","2022-01-12 02:32:30.209790","2022-01-11 23:27:53.286713","2022-01-11 23:27:53.286713","2022-01-14 02:59:36.503497","2022-01-14 05:30:39.440559","2022-01-14 05:30:39.440559","2022-01-14 02:59:36.503497"],"y":[29.8,29.8,30.2,30.2,null,15.8,15.8,16.2,16.2,null,5.8,5.8,6.2,6.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(84, 181, 26)","hoverinfo":"name","legendgroup":"rgb(84, 181, 26)","mode":"none","name":"Job 58","showlegend":true,"x":["2022-01-01 14:50:00","2022-01-01 20:25:39.860140","2022-01-01 20:25:39.860140","2022-01-01 14:50:00","2022-01-01 14:50:00","2022-01-01 20:31:39.860140","2022-01-01 21:55:34.825175","2022-01-01 21:55:34.825175","2022-01-01 20:31:39.860140","2022-01-01 20:31:39.860140","2022-01-11 22:18:21.818182","2022-01-12 00:24:14.265734","2022-01-12 00:24:14.265734","2022-01-11 22:18:21.818182","2022-01-11 22:18:21.818182","2022-01-12 00:30:14.265734","2022-01-12 01:37:22.237762","2022-01-12 01:37:22.237762","2022-01-12 00:30:14.265734"],"y":[27.8,27.8,28.2,28.2,null,20.8,20.8,21.2,21.2,null,3.8,3.8,4.2,4.2,null,19.8,19.8,20.2,20.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(85, 129, 26)","hoverinfo":"name","legendgroup":"rgb(85, 129, 26)","mode":"none","name":"Job 85","showlegend":true,"x":["2022-01-01 23:13:29.790210","2022-01-02 01:02:35.244755","2022-01-02 01:02:35.244755","2022-01-01 23:13:29.790210","2022-01-01 23:13:29.790210","2022-01-03 11:01:44.895105","2022-01-03 12:08:52.867133","2022-01-03 12:08:52.867133","2022-01-03 11:01:44.895105","2022-01-03 11:01:44.895105","2022-01-13 16:35:01.258741","2022-01-13 17:16:58.741259","2022-01-13 17:16:58.741259","2022-01-13 16:35:01.258741","2022-01-13 16:35:01.258741","2022-01-19 18:14:17.622378","2022-01-19 18:56:15.104895","2022-01-19 18:56:15.104895","2022-01-19 18:14:17.622378"],"y":[30.8,30.8,31.2,31.2,null,24.8,24.8,25.2,25.2,null,20.8,20.8,21.2,21.2,null,6.8,6.8,7.2,7.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(87, 127, 28)","hoverinfo":"name","legendgroup":"rgb(87, 127, 28)","mode":"none","name":"Job 48","showlegend":true,"x":["2022-01-03 06:41:36.503497","2022-01-03 07:23:33.986014","2022-01-03 07:23:33.986014","2022-01-03 06:41:36.503497","2022-01-03 06:41:36.503497","2022-01-13 23:11:49.090909","2022-01-14 00:18:57.062937","2022-01-14 00:18:57.062937","2022-01-13 23:11:49.090909","2022-01-13 23:11:49.090909","2022-02-05 02:49:32.307692","2022-02-05 03:31:29.790210","2022-02-05 03:31:29.790210","2022-02-05 02:49:32.307692","2022-02-05 02:49:32.307692","2022-02-05 03:37:29.790210","2022-02-05 04:19:27.272727","2022-02-05 04:19:27.272727","2022-02-05 03:37:29.790210"],"y":[26.8,26.8,27.2,27.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2,null,0.8,0.8,1.2,1.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(87, 30, 194)","hoverinfo":"name","legendgroup":"rgb(87, 30, 194)","mode":"none","name":"Job 76","showlegend":true,"x":["2022-01-01 15:57:07.972028","2022-01-01 21:32:47.832168","2022-01-01 21:32:47.832168","2022-01-01 15:57:07.972028","2022-01-01 15:57:07.972028","2022-01-09 22:56:00","2022-01-10 00:19:54.965035","2022-01-10 00:19:54.965035","2022-01-09 22:56:00","2022-01-09 22:56:00","2022-01-13 18:01:19.720280","2022-01-13 19:42:01.678322","2022-01-13 19:42:01.678322","2022-01-13 18:01:19.720280","2022-01-13 18:01:19.720280","2022-01-19 12:15:50.769231","2022-01-19 13:22:58.741259","2022-01-19 13:22:58.741259","2022-01-19 12:15:50.769231"],"y":[22.8,22.8,23.2,23.2,null,31.8,31.8,32.2,32.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(9, 205, 17)","hoverinfo":"name","legendgroup":"rgb(9, 205, 17)","mode":"none","name":"Job 3","showlegend":true,"x":["2022-01-01 18:44:57.902098","2022-01-01 19:52:05.874126","2022-01-01 19:52:05.874126","2022-01-01 18:44:57.902098","2022-01-01 18:44:57.902098","2022-01-13 08:55:52.447552","2022-01-13 09:37:49.930070","2022-01-13 09:37:49.930070","2022-01-13 08:55:52.447552","2022-01-13 08:55:52.447552","2022-01-13 10:56:57.902098","2022-01-13 11:22:08.391608","2022-01-13 11:22:08.391608","2022-01-13 10:56:57.902098","2022-01-13 10:56:57.902098","2022-01-13 09:43:49.930070","2022-01-13 10:50:57.902098","2022-01-13 10:50:57.902098","2022-01-13 09:43:49.930070"],"y":[31.8,31.8,32.2,32.2,null,30.8,30.8,31.2,31.2,null,2.8,2.8,3.2,3.2,null,31.8,31.8,32.2,32.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(9, 207, 225)","hoverinfo":"name","legendgroup":"rgb(9, 207, 225)","mode":"none","name":"Job 51","showlegend":true,"x":["2022-01-03 01:05:56.643357","2022-01-03 02:46:38.601399","2022-01-03 02:46:38.601399","2022-01-03 01:05:56.643357","2022-01-03 01:05:56.643357","2022-01-13 19:42:01.678322","2022-01-13 21:47:54.125874","2022-01-13 21:47:54.125874","2022-01-13 19:42:01.678322"],"y":[26.8,26.8,27.2,27.2,null,30.8,30.8,31.2,31.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(9, 94, 130)","hoverinfo":"name","legendgroup":"rgb(9, 94, 130)","mode":"none","name":"Job 73","showlegend":true,"x":["2022-01-02 09:00:54.545455","2022-01-02 17:24:24.335664","2022-01-02 17:24:24.335664","2022-01-02 09:00:54.545455","2022-01-02 09:00:54.545455","2022-01-10 14:04:41.118881","2022-01-11 01:16:00.839161","2022-01-11 01:16:00.839161","2022-01-10 14:04:41.118881","2022-01-10 14:04:41.118881","2022-01-11 01:22:00.839161","2022-01-11 02:45:55.804196","2022-01-11 02:45:55.804196","2022-01-11 01:22:00.839161","2022-01-11 01:22:00.839161","2022-01-11 02:51:55.804196","2022-01-11 07:03:40.699301","2022-01-11 07:03:40.699301","2022-01-11 02:51:55.804196","2022-01-11 02:51:55.804196","2022-01-13 16:43:24.755245","2022-01-13 17:50:32.727273","2022-01-13 17:50:32.727273","2022-01-13 16:43:24.755245","2022-01-13 16:43:24.755245","2022-01-13 15:13:29.790210","2022-01-13 16:37:24.755245","2022-01-13 16:37:24.755245","2022-01-13 15:13:29.790210"],"y":[17.8,17.8,18.2,18.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2,null,6.8,6.8,7.2,7.2,null,26.8,26.8,27.2,27.2,null,30.8,30.8,31.2,31.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(90, 88, 8)","hoverinfo":"name","legendgroup":"rgb(90, 88, 8)","mode":"none","name":"Job 25","showlegend":true,"x":["2022-01-06 02:39:55.804196","2022-01-06 07:58:48.671329","2022-01-06 07:58:48.671329","2022-01-06 02:39:55.804196","2022-01-06 02:39:55.804196","2022-01-06 08:04:48.671329","2022-01-06 09:45:30.629371","2022-01-06 09:45:30.629371","2022-01-06 08:04:48.671329","2022-01-06 08:04:48.671329","2022-01-09 10:39:55.804196","2022-01-09 12:45:48.251748","2022-01-09 12:45:48.251748","2022-01-09 10:39:55.804196","2022-01-09 10:39:55.804196","2022-01-09 18:52:38.601399","2022-01-09 20:16:33.566434","2022-01-09 20:16:33.566434","2022-01-09 18:52:38.601399","2022-01-09 18:52:38.601399","2022-01-10 00:34:18.461538","2022-01-10 04:37:39.860140","2022-01-10 04:37:39.860140","2022-01-10 00:34:18.461538"],"y":[27.8,27.8,28.2,28.2,null,15.8,15.8,16.2,16.2,null,30.8,30.8,31.2,31.2,null,12.8,12.8,13.2,13.2,null,4.8,4.8,5.2,5.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(91, 129, 170)","hoverinfo":"name","legendgroup":"rgb(91, 129, 170)","mode":"none","name":"Job 17","showlegend":true,"x":["2022-01-01 19:01:44.895105","2022-01-01 21:49:34.825175","2022-01-01 21:49:34.825175","2022-01-01 19:01:44.895105","2022-01-01 19:01:44.895105","2022-01-01 21:55:34.825175","2022-01-01 23:02:42.797203","2022-01-01 23:02:42.797203","2022-01-01 21:55:34.825175","2022-01-01 21:55:34.825175","2022-01-14 15:45:38.181818","2022-01-14 16:27:35.664336","2022-01-14 16:27:35.664336","2022-01-14 15:45:38.181818","2022-01-14 15:45:38.181818","2022-01-14 16:33:35.664336","2022-01-14 18:14:17.622378","2022-01-14 18:14:17.622378","2022-01-14 16:33:35.664336"],"y":[30.8,30.8,31.2,31.2,null,15.8,15.8,16.2,16.2,null,0.8,0.8,1.2,1.2,null,24.8,24.8,25.2,25.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(93, 30, 135)","hoverinfo":"name","legendgroup":"rgb(93, 30, 135)","mode":"none","name":"Job 49","showlegend":true,"x":["2022-01-03 03:53:46.573427","2022-01-03 05:17:41.538462","2022-01-03 05:17:41.538462","2022-01-03 03:53:46.573427","2022-01-03 03:53:46.573427","2022-01-11 17:04:15.944056","2022-01-11 17:46:13.426573","2022-01-11 17:46:13.426573","2022-01-11 17:04:15.944056","2022-01-11 17:04:15.944056","2022-01-18 06:45:13.006993","2022-01-18 07:27:10.489510","2022-01-18 07:27:10.489510","2022-01-18 06:45:13.006993","2022-01-18 06:45:13.006993","2022-01-18 07:33:10.489510","2022-01-18 11:44:55.384615","2022-01-18 11:44:55.384615","2022-01-18 07:33:10.489510"],"y":[26.8,26.8,27.2,27.2,null,30.8,30.8,31.2,31.2,null,20.8,20.8,21.2,21.2,null,31.8,31.8,32.2,32.2],"type":"scatter"},{"fill":"toself","fillcolor":"rgb(96, 18, 240)","hoverinfo":"name","legendgroup":"rgb(96, 18, 240)","mode":"none","name":"Job 119","showlegend":true,"x":["2022-01-08 08:53:28.951049","2022-01-17 02:40:53.706294","2022-01-17 02:40:53.706294","2022-01-08 08:53:28.951049","2022-01-08 08:53:28.951049","2022-01-17 02:46:53.706294","2022-01-18 06:45:13.006993","2022-01-18 06:45:13.006993","2022-01-17 02:46:53.706294","2022-01-17 02:46:53.706294","2022-01-18 06:51:13.006993","2022-02-01 06:31:04.615385","2022-02-01 06:31:04.615385","2022-01-18 06:51:13.006993"],"y":[8.8,8.8,9.2,9.2,null,20.8,20.8,21.2,21.2,null,14.8,14.8,15.2,15.2],"type":"scatter"},{"legendgroup":"rgb(0, 31, 43)","marker":{"color":"rgb(0, 31, 43)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-03 00:23:59.160839","2022-01-03 02:29:51.608392","2022-01-11 21:41:11.328671","2022-01-11 23:05:06.293706","2022-01-11 23:11:06.293706","2022-01-11 23:27:53.286713","2022-01-11 23:33:53.286713","2022-01-11 23:59:03.776224"],"y":[19,19,31,31,16,16,7,7],"type":"scatter"},{"legendgroup":"rgb(1, 19, 61)","marker":{"color":"rgb(1, 19, 61)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-03 02:46:38.601399","2022-01-03 03:53:46.573427","2022-01-13 22:04:41.118881","2022-01-14 00:10:33.566434","2022-01-14 05:33:02.937063","2022-01-14 06:40:10.909091"],"y":[27,27,26,26,1,1],"type":"scatter"},{"legendgroup":"rgb(100, 36, 174)","marker":{"color":"rgb(100, 36, 174)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-02 09:48:52.027972","2022-01-14 08:31:39.860140","2022-02-03 02:59:36.503497","2022-02-04 20:57:05.454545"],"y":[8,8,21,21],"type":"scatter"},{"legendgroup":"rgb(103, 188, 246)","marker":{"color":"rgb(103, 188, 246)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 15:57:07.972028","2022-01-11 01:16:00.839161","2022-01-11 02:39:55.804196","2022-02-03 02:17:39.020979","2022-02-03 02:59:36.503497","2022-02-03 03:05:36.503497","2022-02-03 03:47:33.986014"],"y":[23,23,31,31,21,21,3,3],"type":"scatter"},{"legendgroup":"rgb(104, 213, 69)","marker":{"color":"rgb(104, 213, 69)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-04 07:35:18.881119","2022-01-05 04:34:03.356643","2022-01-05 04:40:03.356643","2022-01-05 07:27:53.286713","2022-01-05 07:33:53.286713","2022-01-06 08:44:22.657343","2022-01-11 20:48:26.853147","2022-01-11 22:12:21.818182"],"y":[28,28,4,4,5,5,16,16],"type":"scatter"},{"legendgroup":"rgb(106, 59, 98)","marker":{"color":"rgb(106, 59, 98)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-03 07:23:33.986014","2022-01-03 09:04:15.944056","2022-01-11 13:51:15.524476","2022-01-11 15:57:07.972028","2022-01-19 05:16:15.944056","2022-01-19 06:23:23.916084"],"y":[23,23,31,31,21,21],"type":"scatter"},{"legendgroup":"rgb(107, 118, 217)","marker":{"color":"rgb(107, 118, 217)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-07 16:42:26.853147","2022-01-08 09:29:26.433566","2022-01-08 09:35:26.433566","2022-01-08 12:23:16.363636","2022-01-08 12:29:16.363636","2022-01-09 10:09:58.321678","2022-01-12 04:13:12.167832","2022-01-12 05:37:07.132867"],"y":[31,31,4,4,0,0,16,16],"type":"scatter"},{"legendgroup":"rgb(11, 153, 214)","marker":{"color":"rgb(11, 153, 214)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 21:49:34.825175","2022-01-09 14:15:43.216783","2022-01-09 16:29:59.160839","2022-01-12 06:50:15.104895","2022-01-12 20:49:24.755245","2022-01-13 09:16:15.944056","2022-01-13 12:04:05.874126"],"y":[8,8,1,1,6,6,29,29],"type":"scatter"},{"legendgroup":"rgb(114, 35, 134)","marker":{"color":"rgb(114, 35, 134)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-01 21:32:47.832168","2022-01-02 00:20:37.762238","2022-01-02 20:37:24.755245","2022-01-02 22:01:19.720280","2022-02-05 19:36:31.888112","2022-02-05 20:18:29.370629"],"y":[23,23,28,28,21,21],"type":"scatter"},{"legendgroup":"rgb(116, 11, 97)","marker":{"color":"rgb(116, 11, 97)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-11 17:46:13.426573","2022-01-11 18:53:21.398601","2022-01-11 18:59:21.398601","2022-01-11 19:58:05.874126","2022-01-19 04:34:18.461538","2022-01-19 05:16:15.944056","2022-01-19 05:22:15.944056","2022-01-19 06:04:13.426573"],"y":[31,31,25,25,21,21,3,3],"type":"scatter"},{"legendgroup":"rgb(116, 89, 192)","marker":{"color":"rgb(116, 89, 192)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 06:29:51.608392","2022-01-02 09:59:39.020979","2022-01-08 10:28:10.909091","2022-01-08 11:35:18.881119","2022-01-11 07:03:40.699301","2022-01-11 08:52:46.153846"],"y":[27,27,26,26,7,7],"type":"scatter"},{"legendgroup":"rgb(12, 159, 73)","marker":{"color":"rgb(12, 159, 73)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-05 03:52:05.874126","2022-01-06 21:49:34.825175","2022-01-06 21:55:34.825175","2022-01-08 01:53:54.125874","2022-01-10 21:33:02.937063","2022-01-13 16:41:01.258741","2022-01-18 11:05:21.398601","2022-01-19 01:04:31.048951"],"y":[26,26,9,9,10,10,21,21],"type":"scatter"},{"legendgroup":"rgb(13, 19, 155)","marker":{"color":"rgb(13, 19, 155)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-09 12:45:48.251748","2022-01-09 13:52:56.223776","2022-01-09 13:58:56.223776","2022-01-09 14:15:43.216783","2022-01-13 16:21:50.769231","2022-01-13 17:45:45.734266"],"y":[31,31,1,1,0,0],"type":"scatter"},{"legendgroup":"rgb(132, 86, 253)","marker":{"color":"rgb(132, 86, 253)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-08 14:39:55.804196","2022-01-08 14:45:55.804196","2022-01-10 08:43:24.755245","2022-01-10 08:49:24.755245","2022-02-07 08:09:07.972028","2022-02-07 08:15:07.972028","2022-02-09 16:11:46.573427","2022-02-09 16:17:46.573427","2022-02-10 13:16:31.048951","2022-02-10 13:22:31.048951","2022-02-11 17:20:50.349650"],"y":[29,29,3,3,2,2,14,14,16,16,13,13],"type":"scatter"},{"legendgroup":"rgb(134, 4, 66)","marker":{"color":"rgb(134, 4, 66)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 19:01:44.895105","2022-01-01 23:02:42.797203","2022-01-02 00:26:37.762238","2022-01-13 11:22:08.391608","2022-01-13 12:46:03.356643","2022-01-13 23:57:23.076923","2022-01-14 02:03:15.524476"],"y":[31,31,16,16,3,3,10,10],"type":"scatter"},{"legendgroup":"rgb(135, 229, 34)","marker":{"color":"rgb(135, 229, 34)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 15:57:07.972028","2022-01-02 09:00:54.545455","2022-01-02 09:42:52.027972","2022-01-18 09:16:15.944056","2022-01-18 10:23:23.916084","2022-01-18 11:44:55.384615","2022-01-18 15:14:42.797203"],"y":[25,25,28,28,21,21,32,32],"type":"scatter"},{"legendgroup":"rgb(136, 135, 160)","marker":{"color":"rgb(136, 135, 160)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 19:01:44.895105","2022-01-01 23:30:16.783217","2022-01-07 19:47:03.776224","2022-01-07 20:54:11.748252","2022-01-09 11:27:53.286713","2022-01-09 12:09:50.769231","2022-01-14 15:03:40.699301","2022-01-14 15:45:38.181818"],"y":[27,27,26,26,21,21,1,1],"type":"scatter"},{"legendgroup":"rgb(137, 176, 240)","marker":{"color":"rgb(137, 176, 240)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 21:49:34.825175","2022-01-01 23:13:29.790210","2022-01-02 19:55:27.272727","2022-01-02 21:02:35.244755","2022-01-13 07:12:46.993007","2022-01-13 07:54:44.475524","2022-01-14 06:40:10.909091","2022-01-14 07:22:08.391608"],"y":[31,31,25,25,16,16,1,1],"type":"scatter"},{"legendgroup":"rgb(14, 242, 197)","marker":{"color":"rgb(14, 242, 197)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-05 01:27:02.937063","2022-01-05 02:50:57.902098","2022-01-09 13:52:56.223776","2022-01-09 14:34:53.706294","2022-01-13 12:46:03.356643","2022-01-13 13:11:13.846154","2022-01-09 14:40:53.706294","2022-01-09 16:04:48.671329"],"y":[32,32,31,31,3,3,32,32],"type":"scatter"},{"legendgroup":"rgb(142, 4, 253)","marker":{"color":"rgb(142, 4, 253)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 23:13:29.790210","2022-01-02 04:49:09.650350","2022-01-08 01:34:43.636364","2022-01-08 08:34:18.461538","2022-01-13 05:48:52.027972","2022-01-13 07:12:46.993007","2022-01-19 18:56:15.104895","2022-01-19 21:02:07.552448"],"y":[17,17,15,15,16,16,7,7],"type":"scatter"},{"legendgroup":"rgb(147, 30, 215)","marker":{"color":"rgb(147, 30, 215)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 01:36:09.230769","2022-01-02 08:18:57.062937","2022-01-02 08:24:57.062937","2022-01-02 09:48:52.027972","2022-01-09 12:09:50.769231","2022-01-09 14:15:43.216783"],"y":[18,18,8,8,21,21],"type":"scatter"},{"legendgroup":"rgb(15, 34, 188)","marker":{"color":"rgb(15, 34, 188)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-08 21:47:54.125874","2022-01-09 01:59:39.020979","2022-01-09 10:03:58.321678","2022-01-09 11:27:53.286713","2022-01-13 15:59:03.776224","2022-01-13 17:22:58.741259","2022-01-13 21:23:56.643357","2022-01-14 02:59:36.503497"],"y":[31,31,21,21,3,3,6,6],"type":"scatter"},{"legendgroup":"rgb(151, 102, 182)","marker":{"color":"rgb(151, 102, 182)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 20:25:39.860140","2022-01-01 21:49:34.825175","2022-01-12 19:30:16.783217","2022-01-12 22:01:19.720280","2022-01-12 22:07:19.720280","2022-01-12 22:49:17.202797","2022-01-12 22:55:17.202797","2022-01-13 00:35:59.160839","2022-01-13 21:09:33.146853","2022-01-13 23:57:23.076923"],"y":[18,18,26,26,16,16,13,13,10,10],"type":"scatter"},{"legendgroup":"rgb(157, 128, 184)","marker":{"color":"rgb(157, 128, 184)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-04 06:28:10.909091","2022-01-04 07:35:18.881119","2022-01-19 06:23:23.916084","2022-01-19 07:05:21.398601","2022-01-19 07:11:21.398601","2022-01-19 08:18:29.370629"],"y":[28,28,21,21,17,17],"type":"scatter"},{"legendgroup":"rgb(157, 145, 15)","marker":{"color":"rgb(157, 145, 15)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 19:01:44.895105","2022-01-06 07:56:25.174825","2022-01-06 09:37:07.132867","2022-01-06 09:43:07.132867","2022-01-06 20:54:26.853147","2022-01-09 02:51:13.006993","2022-01-09 05:39:02.937063"],"y":[15,15,7,7,11,11,29,29],"type":"scatter"},{"legendgroup":"rgb(162, 107, 85)","marker":{"color":"rgb(162, 107, 85)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-04 11:21:53.286713","2022-01-04 13:02:35.244755","2022-01-14 00:18:57.062937","2022-01-14 01:42:52.027972","2022-02-05 20:18:29.370629","2022-02-05 21:00:26.853147"],"y":[20,20,31,31,21,21],"type":"scatter"},{"legendgroup":"rgb(162, 2, 232)","marker":{"color":"rgb(162, 2, 232)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 01:44:32.727273","2022-01-03 05:42:52.027972","2022-01-11 23:05:06.293706","2022-01-13 05:51:15.524476","2022-02-05 06:19:19.720280","2022-02-05 17:30:39.440559"],"y":[23,23,31,31,21,21],"type":"scatter"},{"legendgroup":"rgb(164, 79, 82)","marker":{"color":"rgb(164, 79, 82)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 19:01:44.895105","2022-01-01 19:52:05.874126","2022-01-01 21:16:00.839161","2022-01-06 21:49:34.825175","2022-01-06 23:13:29.790210","2022-01-11 18:00:36.923077","2022-01-11 19:07:44.895105"],"y":[27,27,32,32,26,26,16,16],"type":"scatter"},{"legendgroup":"rgb(165, 79, 195)","marker":{"color":"rgb(165, 79, 195)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-13 07:15:10.489510","2022-01-13 08:05:31.468531","2022-01-13 08:11:31.468531","2022-01-13 08:28:18.461538","2022-01-13 16:41:01.258741","2022-01-13 17:39:45.734266"],"y":[31,31,3,3,10,10],"type":"scatter"},{"legendgroup":"rgb(167, 87, 72)","marker":{"color":"rgb(167, 87, 72)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-05 02:44:57.902098","2022-01-05 03:52:05.874126","2022-01-06 09:45:30.629371","2022-01-06 10:10:41.118881"],"y":[26,26,16,16],"type":"scatter"},{"legendgroup":"rgb(170, 187, 162)","marker":{"color":"rgb(170, 187, 162)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-11 01:07:37.342657","2022-01-11 06:43:17.202797","2022-01-11 12:33:20.559441","2022-01-11 16:03:07.972028","2022-01-11 16:09:07.972028","2022-01-12 00:32:37.762238"],"y":[26,26,1,1,6,6],"type":"scatter"},{"legendgroup":"rgb(176, 205, 241)","marker":{"color":"rgb(176, 205, 241)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 16:00:29.370629","2022-01-03 06:41:36.503497","2022-01-09 10:45:55.804196","2022-01-09 12:26:37.762238","2022-01-09 12:32:37.762238","2022-01-10 07:50:40.279720"],"y":[14,14,16,16,29,29],"type":"scatter"},{"legendgroup":"rgb(176, 213, 94)","marker":{"color":"rgb(176, 213, 94)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 20:25:39.860140","2022-01-06 08:40:46.153846","2022-01-06 10:21:28.111888","2022-01-18 07:27:10.489510","2022-01-18 08:09:07.972028"],"y":[18,18,28,28,21,21],"type":"scatter"},{"legendgroup":"rgb(178, 127, 166)","marker":{"color":"rgb(178, 127, 166)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 00:20:37.762238","2022-01-02 01:44:32.727273","2022-01-11 06:43:17.202797","2022-01-11 08:49:09.650350","2022-01-11 08:55:09.650350","2022-01-11 10:02:17.622378"],"y":[23,23,26,26,1,1],"type":"scatter"},{"legendgroup":"rgb(179, 168, 226)","marker":{"color":"rgb(179, 168, 226)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-01 17:21:02.937063","2022-01-01 18:44:57.902098","2022-01-14 04:22:18.461538","2022-01-14 05:46:13.426573"],"y":[32,32,26,26],"type":"scatter"},{"legendgroup":"rgb(179, 202, 141)","marker":{"color":"rgb(179, 202, 141)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-03 06:41:36.503497","2022-01-03 23:28:36.083916","2022-01-12 02:32:30.209790","2022-01-12 04:13:12.167832","2022-01-13 12:04:05.874126","2022-01-14 16:02:25.174825"],"y":[14,14,16,16,29,29],"type":"scatter"},{"legendgroup":"rgb(179, 45, 223)","marker":{"color":"rgb(179, 45, 223)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-02 18:48:19.300699","2022-01-05 09:52:56.223776","2022-01-05 23:52:05.874126","2022-01-09 14:15:43.216783","2022-01-09 21:15:18.041958","2022-01-14 08:04:05.874126","2022-01-14 15:03:40.699301"],"y":[19,19,28,28,21,21,1,1],"type":"scatter"},{"legendgroup":"rgb(18, 187, 67)","marker":{"color":"rgb(18, 187, 67)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 16:13:54.965035","2022-01-14 00:10:33.566434","2022-01-14 02:16:26.013986"],"y":[17,17,26,26],"type":"scatter"},{"legendgroup":"rgb(18, 248, 215)","marker":{"color":"rgb(18, 248, 215)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-08 11:35:18.881119","2022-01-10 23:43:42.377622","2022-01-13 11:07:44.895105","2022-01-13 15:19:29.790210"],"y":[26,26,16,16],"type":"scatter"},{"legendgroup":"rgb(186, 32, 212)","marker":{"color":"rgb(186, 32, 212)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-03 09:04:15.944056","2022-01-03 11:35:18.881119","2022-01-13 09:37:49.930070","2022-01-13 11:01:44.895105","2022-02-04 22:21:00.419580","2022-02-04 23:02:57.902098"],"y":[23,23,31,31,21,21],"type":"scatter"},{"legendgroup":"rgb(187, 19, 50)","marker":{"color":"rgb(187, 19, 50)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-03 07:23:33.986014","2022-01-03 08:05:31.468531","2022-01-13 08:05:31.468531","2022-01-13 08:55:52.447552","2022-02-05 17:30:39.440559","2022-02-05 18:12:36.923077","2022-02-05 18:18:36.923077","2022-02-05 19:00:34.405594"],"y":[27,27,31,31,21,21,7,7],"type":"scatter"},{"legendgroup":"rgb(189, 21, 77)","marker":{"color":"rgb(189, 21, 77)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-03 04:35:44.055944","2022-01-03 07:23:33.986014","2022-01-14 02:16:26.013986","2022-01-14 04:22:18.461538"],"y":[18,18,26,26],"type":"scatter"},{"legendgroup":"rgb(191, 6, 147)","marker":{"color":"rgb(191, 6, 147)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-08 04:18:57.062937","2022-01-08 10:36:34.405594","2022-01-09 08:40:03.356643","2022-01-09 10:03:58.321678","2022-01-13 06:11:39.020979","2022-01-13 08:59:28.951049","2022-01-13 09:05:28.951049","2022-01-13 10:12:36.923077"],"y":[28,28,21,21,13,13,19,19],"type":"scatter"},{"legendgroup":"rgb(191, 80, 121)","marker":{"color":"rgb(191, 80, 121)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 20:25:39.860140","2022-01-01 20:31:39.860140","2022-01-03 11:41:18.881119","2022-01-03 11:47:18.881119","2022-01-03 18:46:53.706294","2022-01-19 01:46:28.531469","2022-01-19 04:34:18.461538","2022-01-19 04:40:18.461538","2022-01-19 06:46:10.909091"],"y":[22,22,11,11,30,30,21,21,7,7],"type":"scatter"},{"legendgroup":"rgb(194, 20, 126)","marker":{"color":"rgb(194, 20, 126)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-07 16:17:16.363636","2022-01-07 18:23:08.811189","2022-01-07 18:29:08.811189","2022-01-07 18:45:55.804196","2022-01-14 02:03:15.524476","2022-01-14 03:10:23.496503"],"y":[26,26,7,7,10,10],"type":"scatter"},{"legendgroup":"rgb(195, 250, 163)","marker":{"color":"rgb(195, 250, 163)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-02 02:18:06.713287","2022-01-03 00:40:46.153846","2022-01-11 13:57:15.524476","2022-01-11 19:32:55.384615"],"y":[30,30,21,21],"type":"scatter"},{"legendgroup":"rgb(195, 45, 130)","marker":{"color":"rgb(195, 45, 130)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-02 22:01:19.720280","2022-01-04 04:47:28.951049","2022-01-04 04:53:28.951049","2022-01-05 17:15:18.041958","2022-01-08 23:34:36.083916","2022-01-09 06:34:10.909091","2022-01-09 12:07:27.272727","2022-01-09 19:07:02.097902","2022-01-14 10:57:55.804196","2022-01-15 12:08:25.174825"],"y":[28,28,10,10,21,21,4,4,10,10],"type":"scatter"},{"legendgroup":"rgb(196, 210, 6)","marker":{"color":"rgb(196, 210, 6)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-11 08:15:35.664336","2022-01-11 13:51:15.524476","2022-01-18 08:51:05.454545","2022-01-18 09:16:15.944056"],"y":[31,31,21,21],"type":"scatter"},{"legendgroup":"rgb(197, 17, 140)","marker":{"color":"rgb(197, 17, 140)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 16:13:54.965035","2022-01-04 11:21:53.286713","2022-01-04 11:27:53.286713","2022-01-05 01:27:02.937063","2022-01-06 10:21:28.111888","2022-01-07 14:19:47.412587","2022-01-11 02:45:55.804196","2022-01-11 13:57:15.524476"],"y":[20,20,32,32,28,28,21,21],"type":"scatter"},{"legendgroup":"rgb(2, 106, 106)","marker":{"color":"rgb(2, 106, 106)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 16:13:54.965035","2022-01-02 01:02:35.244755","2022-01-02 01:44:32.727273","2022-01-08 17:08:35.244755","2022-01-08 17:58:56.223776","2022-01-18 15:14:42.797203","2022-01-18 15:56:40.279720","2022-01-08 18:04:56.223776","2022-01-08 18:46:53.706294"],"y":[20,20,31,31,16,16,32,32,20,20],"type":"scatter"},{"legendgroup":"rgb(2, 227, 92)","marker":{"color":"rgb(2, 227, 92)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null,null,null],"x":["2022-01-02 04:49:09.650350","2022-01-05 09:44:32.727273","2022-01-05 23:52:05.874126","2022-01-05 23:52:05.874126","2022-01-08 20:46:46.153846","2022-01-09 10:45:55.804196","2022-01-09 21:23:41.538462","2022-01-11 12:33:20.559441","2022-01-13 08:05:31.468531","2022-01-13 22:04:41.118881","2022-01-13 22:10:41.118881","2022-01-14 09:22:00.839161"],"y":[17,17,28,28,16,16,13,13,26,26,23,23],"type":"scatter"},{"legendgroup":"rgb(201, 134, 35)","marker":{"color":"rgb(201, 134, 35)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-01 15:31:57.482517","2022-01-01 18:19:47.412587","2022-01-13 05:51:15.524476","2022-01-13 07:15:10.489510","2022-01-18 08:09:07.972028","2022-01-18 08:51:05.454545"],"y":[24,24,31,31,21,21],"type":"scatter"},{"legendgroup":"rgb(205, 226, 10)","marker":{"color":"rgb(205, 226, 10)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-02 02:01:19.720280","2022-01-02 09:00:54.545455","2022-01-05 09:58:56.223776","2022-01-05 11:39:38.181818","2022-01-11 18:53:21.398601","2022-01-11 21:41:11.328671","2022-01-11 21:47:11.328671","2022-01-11 23:11:06.293706","2022-01-14 05:30:39.440559","2022-01-14 11:06:19.300699"],"y":[28,28,16,16,31,31,7,7,6,6],"type":"scatter"},{"legendgroup":"rgb(207, 105, 64)","marker":{"color":"rgb(207, 105, 64)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 18:19:47.412587","2022-01-02 02:43:17.202797","2022-01-02 15:01:44.895105","2022-01-02 20:37:24.755245","2022-01-08 17:58:56.223776","2022-01-08 20:46:46.153846","2022-01-08 20:52:46.153846","2022-01-08 23:57:23.076923","2022-01-09 00:03:23.076923","2022-01-09 02:51:13.006993"],"y":[24,24,28,28,16,16,7,7,29,29],"type":"scatter"},{"legendgroup":"rgb(209, 21, 165)","marker":{"color":"rgb(209, 21, 165)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-08 14:45:55.804196","2022-01-09 10:20:45.314685","2022-01-19 13:22:58.741259","2022-01-19 16:52:46.153846"],"y":[22,22,21,21],"type":"scatter"},{"legendgroup":"rgb(210, 75, 89)","marker":{"color":"rgb(210, 75, 89)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 16:13:54.965035","2022-01-01 23:13:29.790210","2022-01-08 12:17:16.363636","2022-01-08 21:47:54.125874","2022-01-08 22:10:41.118881","2022-01-08 23:34:36.083916","2022-01-13 00:35:59.160839","2022-01-13 04:05:46.573427","2022-01-13 18:07:19.720280","2022-01-13 19:14:27.692308","2022-01-13 16:37:24.755245","2022-01-13 18:01:19.720280"],"y":[17,17,31,31,21,21,13,13,18,18,31,31],"type":"scatter"},{"legendgroup":"rgb(215, 81, 4)","marker":{"color":"rgb(215, 81, 4)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-04 18:46:38.601399","2022-01-07 16:42:26.853147","2022-01-07 16:48:26.853147","2022-01-08 20:46:46.153846","2022-01-08 20:52:46.153846","2022-01-09 10:51:55.804196","2022-01-09 10:57:55.804196","2022-01-11 21:42:24.335664"],"y":[31,31,21,21,1,1,11,11],"type":"scatter"},{"legendgroup":"rgb(216, 241, 158)","marker":{"color":"rgb(216, 241, 158)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 04:57:33.146853","2022-01-03 13:07:37.342657","2022-01-06 23:13:29.790210","2022-01-07 13:12:39.440559","2022-01-12 22:49:17.202797","2022-01-13 05:48:52.027972"],"y":[24,24,26,26,16,16],"type":"scatter"},{"legendgroup":"rgb(22, 148, 1)","marker":{"color":"rgb(22, 148, 1)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-09 21:15:18.041958","2022-01-09 22:56:00","2022-01-13 11:01:44.895105","2022-01-13 12:25:39.860140"],"y":[32,32,31,31],"type":"scatter"},{"legendgroup":"rgb(22, 189, 194)","marker":{"color":"rgb(22, 189, 194)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 23:55:27.272727","2022-01-02 01:36:09.230769","2022-01-07 18:23:08.811189","2022-01-07 19:47:03.776224","2022-01-18 10:23:23.916084","2022-01-18 11:05:21.398601","2022-01-18 11:11:21.398601","2022-01-18 11:53:18.881119"],"y":[18,18,26,26,21,21,13,13],"type":"scatter"},{"legendgroup":"rgb(22, 37, 0)","marker":{"color":"rgb(22, 37, 0)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 01:44:32.727273","2022-01-04 16:40:46.153846","2022-01-04 16:46:46.153846","2022-01-06 07:56:25.174825","2022-01-06 08:02:25.174825","2022-01-10 09:56:32.727273"],"y":[31,31,7,7,12,12],"type":"scatter"},{"legendgroup":"rgb(220, 119, 34)","marker":{"color":"rgb(220, 119, 34)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-09 18:10:41.118881","2022-01-09 21:15:18.041958","2022-01-13 21:47:54.125874","2022-01-13 23:11:49.090909"],"y":[32,32,31,31],"type":"scatter"},{"legendgroup":"rgb(220, 38, 126)","marker":{"color":"rgb(220, 38, 126)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-02 08:18:57.062937","2022-01-02 13:12:39.440559","2022-01-10 15:34:36.083916","2022-01-10 16:41:44.055944"],"y":[9,9,16,16],"type":"scatter"},{"legendgroup":"rgb(224, 24, 147)","marker":{"color":"rgb(224, 24, 147)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-02 14:11:23.916084","2022-01-02 18:23:08.811189","2022-01-08 01:28:43.636364","2022-01-08 06:22:26.013986","2022-01-12 05:37:07.132867","2022-01-12 06:44:15.104895","2022-01-12 06:50:15.104895","2022-01-12 08:30:57.062937"],"y":[27,27,22,22,16,16,1,1],"type":"scatter"},{"legendgroup":"rgb(225, 120, 131)","marker":{"color":"rgb(225, 120, 131)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-09 06:11:23.916084","2022-01-09 09:32:47.832168","2022-01-09 09:38:47.832168","2022-01-09 11:44:40.279720","2022-01-13 13:00:26.853147","2022-01-13 21:23:56.643357","2022-02-05 18:12:36.923077","2022-02-05 19:36:31.888112"],"y":[31,31,30,30,6,6,21,21],"type":"scatter"},{"legendgroup":"rgb(225, 224, 138)","marker":{"color":"rgb(225, 224, 138)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-01 19:01:44.895105","2022-01-01 23:13:29.790210","2022-01-09 06:34:10.909091","2022-01-09 07:58:05.874126","2022-01-09 08:04:05.874126","2022-01-09 09:53:11.328671"],"y":[15,15,21,21,5,5],"type":"scatter"},{"legendgroup":"rgb(225, 62, 236)","marker":{"color":"rgb(225, 62, 236)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-10 11:33:38.181818","2022-01-10 14:04:41.118881","2022-01-10 14:10:41.118881","2022-01-10 14:52:38.601399","2022-01-10 14:58:38.601399","2022-01-10 16:05:46.573427"],"y":[31,31,16,16,18,18],"type":"scatter"},{"legendgroup":"rgb(227, 138, 198)","marker":{"color":"rgb(227, 138, 198)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-02 01:36:09.230769","2022-01-02 03:42:01.678322","2022-01-05 23:52:05.874126","2022-01-06 02:39:55.804196","2022-01-10 14:52:38.601399","2022-01-10 15:34:36.083916","2022-01-14 03:52:20.979021","2022-01-14 05:33:02.937063","2022-01-14 05:39:02.937063","2022-01-14 08:26:52.867133"],"y":[27,27,28,28,16,16,1,1,0,0],"type":"scatter"},{"legendgroup":"rgb(227, 235, 11)","marker":{"color":"rgb(227, 235, 11)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 18:48:19.300699","2022-01-03 00:23:59.160839","2022-01-13 12:25:39.860140","2022-01-13 15:13:29.790210","2022-01-19 21:02:07.552448","2022-01-19 22:51:13.006993"],"y":[19,19,31,31,7,7],"type":"scatter"},{"legendgroup":"rgb(228, 240, 124)","marker":{"color":"rgb(228, 240, 124)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 19:18:31.888112","2022-01-12 05:23:56.643357","2022-01-13 13:34:00.839161","2022-01-13 13:40:00.839161","2022-01-13 19:15:40.699301","2022-02-05 04:13:27.272727","2022-02-05 06:19:19.720280","2022-02-05 06:25:19.720280","2022-02-05 08:31:12.167832"],"y":[30,30,0,0,22,22,21,21,3,3],"type":"scatter"},{"legendgroup":"rgb(231, 216, 25)","marker":{"color":"rgb(231, 216, 25)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-08 01:05:56.643357","2022-01-08 04:10:33.566434","2022-01-08 12:48:26.853147","2022-01-08 13:55:34.825175","2022-01-11 19:41:18.881119","2022-01-11 21:05:13.846154","2022-01-11 21:11:13.846154","2022-01-11 22:51:55.804196"],"y":[26,26,16,16,3,3,24,24],"type":"scatter"},{"legendgroup":"rgb(234, 202, 35)","marker":{"color":"rgb(234, 202, 35)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 23:13:29.790210","2022-01-01 23:19:29.790210","2022-01-04 21:15:18.041958","2022-01-04 21:21:18.041958","2022-01-05 03:38:55.384615","2022-01-08 14:37:32.307692","2022-01-08 17:08:35.244755","2022-01-13 04:05:46.573427","2022-01-13 06:11:39.020979"],"y":[9,9,5,5,15,15,16,16,13,13],"type":"scatter"},{"legendgroup":"rgb(235, 253, 30)","marker":{"color":"rgb(235, 253, 30)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-01 23:30:16.783217","2022-01-02 01:36:09.230769","2022-01-08 09:29:26.433566","2022-01-08 12:17:16.363636","2022-01-13 15:19:29.790210","2022-01-13 16:26:37.762238"],"y":[27,27,31,31,16,16],"type":"scatter"},{"legendgroup":"rgb(235, 37, 200)","marker":{"color":"rgb(235, 37, 200)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 09:59:39.020979","2022-01-02 14:11:23.916084","2022-01-10 23:43:42.377622","2022-01-11 01:07:37.342657","2022-01-11 17:01:52.447552","2022-01-11 18:50:57.902098"],"y":[27,27,26,26,13,13],"type":"scatter"},{"legendgroup":"rgb(242, 153, 176)","marker":{"color":"rgb(242, 153, 176)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-02 21:02:35.244755","2022-01-03 11:01:44.895105","2022-01-09 14:34:53.706294","2022-01-10 11:33:38.181818","2022-01-11 23:59:03.776224","2022-01-12 11:10:23.496503"],"y":[25,25,31,31,7,7],"type":"scatter"},{"legendgroup":"rgb(243, 197, 144)","marker":{"color":"rgb(243, 197, 144)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-02 08:18:57.062937","2022-01-02 09:00:54.545455","2022-01-12 22:01:19.720280","2022-01-12 23:25:14.685315","2022-01-13 08:45:05.454545","2022-01-13 09:01:52.447552","2022-01-19 06:04:13.426573","2022-01-19 06:29:23.916084"],"y":[18,18,26,26,16,16,3,3],"type":"scatter"},{"legendgroup":"rgb(245, 29, 232)","marker":{"color":"rgb(245, 29, 232)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-05 03:38:55.384615","2022-01-08 01:34:43.636364","2022-01-10 08:43:24.755245","2022-01-11 19:41:18.881119","2022-01-11 19:47:18.881119","2022-01-20 13:34:43.636364","2022-01-20 13:40:43.636364","2022-01-21 17:39:02.937063"],"y":[15,15,3,3,5,5,30,30],"type":"scatter"},{"legendgroup":"rgb(246, 74, 41)","marker":{"color":"rgb(246, 74, 41)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 17:21:02.937063","2022-01-06 07:58:48.671329","2022-01-06 08:40:46.153846","2022-01-12 11:10:23.496503","2022-01-12 11:35:33.986014","2022-01-09 16:04:48.671329","2022-01-09 18:10:41.118881"],"y":[32,32,28,28,7,7,32,32],"type":"scatter"},{"legendgroup":"rgb(247, 251, 13)","marker":{"color":"rgb(247, 251, 13)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-03 07:23:33.986014","2022-01-04 04:22:18.461538","2022-01-13 01:05:56.643357","2022-01-13 08:05:31.468531","2022-01-13 08:11:31.468531","2022-01-13 16:35:01.258741","2022-01-13 16:41:01.258741","2022-01-14 03:52:20.979021"],"y":[18,18,26,26,21,21,1,1],"type":"scatter"},{"legendgroup":"rgb(249, 156, 65)","marker":{"color":"rgb(249, 156, 65)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-08 10:36:34.405594","2022-01-09 07:35:18.881119","2022-01-09 07:41:18.881119","2022-01-09 18:52:38.601399","2022-01-09 18:58:38.601399","2022-01-10 21:33:02.937063","2022-01-12 22:24:06.713287","2022-01-13 08:11:31.468531"],"y":[28,28,13,13,10,10,21,21],"type":"scatter"},{"legendgroup":"rgb(252, 125, 21)","marker":{"color":"rgb(252, 125, 21)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-09 01:59:39.020979","2022-01-09 06:11:23.916084","2022-01-12 21:16:58.741259","2022-01-12 22:24:06.713287","2022-01-13 13:11:13.846154","2022-01-13 15:59:03.776224","2022-01-13 17:39:45.734266","2022-01-13 21:09:33.146853"],"y":[31,31,21,21,3,3,10,10],"type":"scatter"},{"legendgroup":"rgb(252, 142, 168)","marker":{"color":"rgb(252, 142, 168)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-14 01:42:52.027972","2022-01-14 06:11:23.916084","2022-01-19 15:09:40.699301","2022-01-19 17:57:30.629371","2022-01-19 18:03:30.629371","2022-01-20 00:21:07.972028"],"y":[31,31,7,7,0,0],"type":"scatter"},{"legendgroup":"rgb(254, 212, 212)","marker":{"color":"rgb(254, 212, 212)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-08 08:34:18.461538","2022-01-08 14:09:58.321678","2022-01-13 09:43:49.930070","2022-01-13 11:07:44.895105","2022-01-14 03:10:23.496503","2022-01-14 05:58:13.426573"],"y":[15,15,16,16,10,10],"type":"scatter"},{"legendgroup":"rgb(255, 1, 148)","marker":{"color":"rgb(255, 1, 148)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-02 09:42:52.027972","2022-01-02 13:54:36.923077","2022-01-09 12:26:37.762238","2022-01-09 13:33:45.734266","2022-01-11 11:09:25.594406","2022-01-11 12:33:20.559441","2022-01-11 12:39:20.559441","2022-01-11 14:20:02.517483"],"y":[28,28,16,16,1,1,17,17],"type":"scatter"},{"legendgroup":"rgb(29, 131, 212)","marker":{"color":"rgb(29, 131, 212)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-03 05:17:41.538462","2022-01-03 06:41:36.503497","2022-01-12 23:25:14.685315","2022-01-13 01:05:56.643357","2022-01-13 07:54:44.475524","2022-01-13 08:45:05.454545","2022-01-14 16:27:35.664336","2022-01-14 17:34:43.636364"],"y":[27,27,26,26,16,16,1,1],"type":"scatter"},{"legendgroup":"rgb(30, 32, 230)","marker":{"color":"rgb(30, 32, 230)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-05 02:44:57.902098","2022-01-05 02:50:57.902098","2022-01-05 23:49:42.377622","2022-01-12 12:53:28.951049","2022-01-13 02:52:38.601399","2022-01-13 02:58:38.601399","2022-01-16 00:54:26.853147","2022-01-11 08:49:09.650350","2022-01-12 12:47:28.951049"],"y":[26,26,21,21,7,7,11,11,26,26],"type":"scatter"},{"legendgroup":"rgb(32, 137, 126)","marker":{"color":"rgb(32, 137, 126)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-08 09:04:15.944056","2022-01-08 10:28:10.909091","2022-01-11 10:02:17.622378","2022-01-11 11:09:25.594406","2022-01-12 03:20:27.692308","2022-01-12 06:50:15.104895","2022-01-19 07:05:21.398601","2022-01-19 08:04:05.874126"],"y":[26,26,1,1,6,6,21,21],"type":"scatter"},{"legendgroup":"rgb(35, 212, 8)","marker":{"color":"rgb(35, 212, 8)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 20:25:39.860140","2022-01-02 02:01:19.720280","2022-01-08 20:46:46.153846","2022-01-08 22:10:41.118881","2022-01-11 15:21:10.489510","2022-01-11 17:01:52.447552","2022-01-12 00:32:37.762238","2022-01-12 03:20:27.692308"],"y":[28,28,21,21,13,13,6,6],"type":"scatter"},{"legendgroup":"rgb(35, 26, 177)","marker":{"color":"rgb(35, 26, 177)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-02 04:49:09.650350","2022-01-02 04:55:09.650350","2022-01-02 09:06:54.545455","2022-01-02 09:12:54.545455","2022-01-04 10:09:58.321678","2022-01-08 06:22:26.013986","2022-01-08 14:45:55.804196","2022-01-19 08:04:05.874126","2022-01-19 10:09:58.321678","2022-01-19 10:15:58.321678","2022-01-19 13:03:48.251748"],"y":[14,14,13,13,12,12,22,22,21,21,1,1],"type":"scatter"},{"legendgroup":"rgb(4, 208, 74)","marker":{"color":"rgb(4, 208, 74)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-02 17:24:24.335664","2022-01-03 04:35:44.055944","2022-01-12 12:47:28.951049","2022-01-12 18:06:21.818182","2022-01-12 18:12:21.818182","2022-01-12 21:16:58.741259","2022-01-13 02:52:38.601399","2022-01-13 06:22:26.013986","2022-01-13 06:28:26.013986","2022-01-13 09:16:15.944056"],"y":[18,18,26,26,21,21,7,7,29,29],"type":"scatter"},{"legendgroup":"rgb(41, 39, 121)","marker":{"color":"rgb(41, 39, 121)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-03 05:42:52.027972","2022-01-03 07:23:33.986014","2022-01-11 15:57:07.972028","2022-01-11 17:04:15.944056","2022-01-11 17:10:15.944056","2022-01-11 18:00:36.923077","2022-01-11 18:06:36.923077","2022-01-11 19:13:44.895105","2022-01-19 06:04:13.426573","2022-01-19 06:46:10.909091"],"y":[23,23,31,31,16,16,27,27,32,32],"type":"scatter"},{"legendgroup":"rgb(46, 247, 226)","marker":{"color":"rgb(46, 247, 226)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-02 18:23:08.811189","2022-01-03 01:05:56.643357","2022-01-08 04:10:33.566434","2022-01-08 09:04:15.944056","2022-01-09 21:15:18.041958","2022-01-09 22:56:00","2022-01-11 12:33:20.559441","2022-01-11 15:21:10.489510","2022-01-11 15:27:10.489510","2022-01-11 17:58:13.426573"],"y":[27,27,26,26,21,21,13,13,15,15],"type":"scatter"},{"legendgroup":"rgb(47, 43, 189)","marker":{"color":"rgb(47, 43, 189)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-04 16:40:46.153846","2022-01-04 18:46:38.601399","2022-01-04 18:52:38.601399","2022-01-04 21:40:28.531469","2022-01-13 17:16:58.741259","2022-01-13 17:58:56.223776","2022-01-14 07:22:08.391608","2022-01-14 08:04:05.874126","2022-01-14 08:10:05.874126","2022-01-14 10:57:55.804196"],"y":[31,31,6,6,21,21,1,1,10,10],"type":"scatter"},{"legendgroup":"rgb(50, 21, 217)","marker":{"color":"rgb(50, 21, 217)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 21:49:34.825175","2022-01-01 23:55:27.272727","2022-01-12 18:06:21.818182","2022-01-12 19:30:16.783217","2022-02-05 03:31:29.790210","2022-02-05 04:13:27.272727","2022-02-05 04:19:27.272727","2022-02-05 05:01:24.755245"],"y":[18,18,26,26,21,21,4,4],"type":"scatter"},{"legendgroup":"rgb(51, 45, 236)","marker":{"color":"rgb(51, 45, 236)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-01 23:13:29.790210","2022-01-02 08:18:57.062937","2022-02-05 00:01:42.377622","2022-02-05 01:25:37.342657"],"y":[9,9,21,21],"type":"scatter"},{"legendgroup":"rgb(52, 113, 143)","marker":{"color":"rgb(52, 113, 143)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-02 03:42:01.678322","2022-01-02 06:29:51.608392","2022-01-07 13:12:39.440559","2022-01-07 16:17:16.363636","2022-01-09 07:58:05.874126","2022-01-09 08:40:03.356643","2022-01-09 08:46:03.356643","2022-01-09 10:26:45.314685","2022-01-13 13:34:00.839161","2022-01-13 16:21:50.769231"],"y":[27,27,26,26,21,21,4,4,0,0],"type":"scatter"},{"legendgroup":"rgb(56, 168, 62)","marker":{"color":"rgb(56, 168, 62)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-07 20:54:11.748252","2022-01-07 23:42:01.678322","2022-01-07 23:48:01.678322","2022-01-08 01:28:43.636364","2022-01-09 10:09:58.321678","2022-01-09 15:45:38.181818","2022-02-04 20:57:05.454545","2022-02-04 22:21:00.419580"],"y":[26,26,22,22,0,0,21,21],"type":"scatter"},{"legendgroup":"rgb(57, 187, 49)","marker":{"color":"rgb(57, 187, 49)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-07 23:42:01.678322","2022-01-08 01:05:56.643357","2022-01-08 13:55:34.825175","2022-01-08 14:37:32.307692","2022-01-08 14:43:32.307692","2022-01-08 15:50:40.279720"],"y":[26,26,16,16,24,24],"type":"scatter"},{"legendgroup":"rgb(57, 228, 159)","marker":{"color":"rgb(57, 228, 159)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-09 07:35:18.881119","2022-01-18 15:21:53.286713","2022-01-18 15:27:53.286713","2022-01-21 13:23:41.538462","2022-01-21 13:29:41.538462","2022-02-02 12:12:29.370629","2022-02-02 12:18:29.370629","2022-02-03 02:17:39.020979"],"y":[28,28,4,4,12,12,21,21],"type":"scatter"},{"legendgroup":"rgb(6, 92, 92)","marker":{"color":"rgb(6, 92, 92)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 15:31:57.482517","2022-01-02 13:54:36.923077","2022-01-02 15:01:44.895105","2022-01-05 11:39:38.181818","2022-01-05 11:56:25.174825","2022-01-19 17:57:30.629371","2022-01-19 18:14:17.622378"],"y":[24,24,28,28,16,16,7,7],"type":"scatter"},{"legendgroup":"rgb(61, 240, 53)","marker":{"color":"rgb(61, 240, 53)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-09 09:32:47.832168","2022-01-09 10:39:55.804196","2022-01-09 20:16:33.566434","2022-01-09 21:23:41.538462","2022-01-09 21:29:41.538462","2022-01-10 00:34:18.461538","2022-02-04 23:02:57.902098","2022-02-05 00:01:42.377622"],"y":[31,31,13,13,5,5,21,21],"type":"scatter"},{"legendgroup":"rgb(63, 8, 11)","marker":{"color":"rgb(63, 8, 11)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-04 04:47:28.951049","2022-01-04 06:28:10.909091","2022-01-04 06:34:10.909091","2022-01-04 09:05:13.846154","2022-01-13 09:01:52.447552","2022-01-13 09:43:49.930070","2022-01-13 09:49:49.930070","2022-01-13 10:31:47.412587","2022-01-13 10:37:47.412587","2022-01-13 13:00:26.853147"],"y":[28,28,2,2,16,16,3,3,6,6],"type":"scatter"},{"legendgroup":"rgb(65, 217, 136)","marker":{"color":"rgb(65, 217, 136)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null,null,null],"x":["2022-01-02 04:49:09.650350","2022-01-02 16:00:29.370629","2022-01-05 07:27:53.286713","2022-01-05 11:39:38.181818","2022-01-05 11:45:38.181818","2022-01-07 02:55:17.202797","2022-01-08 14:09:58.321678","2022-01-08 19:45:38.181818","2022-01-19 10:09:58.321678","2022-01-19 12:15:50.769231","2022-01-19 12:21:50.769231","2022-01-19 15:09:40.699301"],"y":[14,14,4,4,6,6,15,15,21,21,7,7],"type":"scatter"},{"legendgroup":"rgb(66, 158, 132)","marker":{"color":"rgb(66, 158, 132)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-08 01:53:54.125874","2022-01-08 08:53:28.951049","2022-02-05 01:25:37.342657","2022-02-05 02:49:32.307692"],"y":[9,9,21,21],"type":"scatter"},{"legendgroup":"rgb(71, 108, 219)","marker":{"color":"rgb(71, 108, 219)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-02 02:43:17.202797","2022-01-02 04:57:33.146853","2022-01-05 08:45:48.251748","2022-01-05 09:52:56.223776","2022-01-19 01:04:31.048951","2022-01-19 01:46:28.531469","2022-01-19 01:52:28.531469","2022-01-19 06:04:13.426573"],"y":[24,24,28,28,21,21,32,32],"type":"scatter"},{"legendgroup":"rgb(71, 247, 110)","marker":{"color":"rgb(71, 247, 110)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-05 04:34:03.356643","2022-01-05 08:45:48.251748","2022-01-05 08:51:48.251748","2022-01-05 09:58:56.223776","2022-01-09 10:26:45.314685","2022-01-09 12:07:27.272727","2022-01-09 12:13:27.272727","2022-01-09 13:20:35.244755"],"y":[28,28,16,16,4,4,24,24],"type":"scatter"},{"legendgroup":"rgb(76, 120, 97)","marker":{"color":"rgb(76, 120, 97)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-01 15:57:07.972028","2022-01-02 19:55:27.272727","2022-01-07 14:19:47.412587","2022-01-08 04:18:57.062937","2022-01-08 04:24:57.062937","2022-01-08 12:48:26.853147","2022-01-08 12:54:26.853147","2022-01-09 00:05:46.573427","2022-01-19 06:46:10.909091","2022-01-19 13:45:45.734266"],"y":[25,25,28,28,16,16,19,19,32,32],"type":"scatter"},{"legendgroup":"rgb(8, 35, 51)","marker":{"color":"rgb(8, 35, 51)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-11 02:39:55.804196","2022-01-11 08:15:35.664336","2022-01-11 19:07:44.895105","2022-01-11 20:48:26.853147","2022-01-11 20:54:26.853147","2022-01-11 22:18:21.818182","2022-01-11 22:24:21.818182","2022-01-12 05:23:56.643357"],"y":[31,31,16,16,4,4,0,0],"type":"scatter"},{"legendgroup":"rgb(80, 199, 222)","marker":{"color":"rgb(80, 199, 222)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-01 19:18:31.888112","2022-01-02 02:18:06.713287","2022-01-11 23:27:53.286713","2022-01-12 02:32:30.209790","2022-01-14 02:59:36.503497","2022-01-14 05:30:39.440559"],"y":[30,30,16,16,6,6],"type":"scatter"},{"legendgroup":"rgb(84, 181, 26)","marker":{"color":"rgb(84, 181, 26)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 14:50:00","2022-01-01 20:25:39.860140","2022-01-01 20:31:39.860140","2022-01-01 21:55:34.825175","2022-01-11 22:18:21.818182","2022-01-12 00:24:14.265734","2022-01-12 00:30:14.265734","2022-01-12 01:37:22.237762"],"y":[28,28,21,21,4,4,20,20],"type":"scatter"},{"legendgroup":"rgb(85, 129, 26)","marker":{"color":"rgb(85, 129, 26)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 23:13:29.790210","2022-01-02 01:02:35.244755","2022-01-03 11:01:44.895105","2022-01-03 12:08:52.867133","2022-01-13 16:35:01.258741","2022-01-13 17:16:58.741259","2022-01-19 18:14:17.622378","2022-01-19 18:56:15.104895"],"y":[31,31,25,25,21,21,7,7],"type":"scatter"},{"legendgroup":"rgb(87, 127, 28)","marker":{"color":"rgb(87, 127, 28)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-03 06:41:36.503497","2022-01-03 07:23:33.986014","2022-01-13 23:11:49.090909","2022-01-14 00:18:57.062937","2022-02-05 02:49:32.307692","2022-02-05 03:31:29.790210","2022-02-05 03:37:29.790210","2022-02-05 04:19:27.272727"],"y":[27,27,31,31,21,21,1,1],"type":"scatter"},{"legendgroup":"rgb(87, 30, 194)","marker":{"color":"rgb(87, 30, 194)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 15:57:07.972028","2022-01-01 21:32:47.832168","2022-01-09 22:56:00","2022-01-10 00:19:54.965035","2022-01-13 18:01:19.720280","2022-01-13 19:42:01.678322","2022-01-19 12:15:50.769231","2022-01-19 13:22:58.741259"],"y":[23,23,32,32,31,31,21,21],"type":"scatter"},{"legendgroup":"rgb(9, 205, 17)","marker":{"color":"rgb(9, 205, 17)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 18:44:57.902098","2022-01-01 19:52:05.874126","2022-01-13 08:55:52.447552","2022-01-13 09:37:49.930070","2022-01-13 10:56:57.902098","2022-01-13 11:22:08.391608","2022-01-13 09:43:49.930070","2022-01-13 10:50:57.902098"],"y":[32,32,31,31,3,3,32,32],"type":"scatter"},{"legendgroup":"rgb(9, 207, 225)","marker":{"color":"rgb(9, 207, 225)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null],"x":["2022-01-03 01:05:56.643357","2022-01-03 02:46:38.601399","2022-01-13 19:42:01.678322","2022-01-13 21:47:54.125874"],"y":[27,27,31,31],"type":"scatter"},{"legendgroup":"rgb(9, 94, 130)","marker":{"color":"rgb(9, 94, 130)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null,null,null],"x":["2022-01-02 09:00:54.545455","2022-01-02 17:24:24.335664","2022-01-10 14:04:41.118881","2022-01-11 01:16:00.839161","2022-01-11 01:22:00.839161","2022-01-11 02:45:55.804196","2022-01-11 02:51:55.804196","2022-01-11 07:03:40.699301","2022-01-13 16:43:24.755245","2022-01-13 17:50:32.727273","2022-01-13 15:13:29.790210","2022-01-13 16:37:24.755245"],"y":[18,18,31,31,21,21,7,7,27,27,31,31],"type":"scatter"},{"legendgroup":"rgb(90, 88, 8)","marker":{"color":"rgb(90, 88, 8)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null,null,null],"x":["2022-01-06 02:39:55.804196","2022-01-06 07:58:48.671329","2022-01-06 08:04:48.671329","2022-01-06 09:45:30.629371","2022-01-09 10:39:55.804196","2022-01-09 12:45:48.251748","2022-01-09 18:52:38.601399","2022-01-09 20:16:33.566434","2022-01-10 00:34:18.461538","2022-01-10 04:37:39.860140"],"y":[28,28,16,16,31,31,13,13,5,5],"type":"scatter"},{"legendgroup":"rgb(91, 129, 170)","marker":{"color":"rgb(91, 129, 170)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-01 19:01:44.895105","2022-01-01 21:49:34.825175","2022-01-01 21:55:34.825175","2022-01-01 23:02:42.797203","2022-01-14 15:45:38.181818","2022-01-14 16:27:35.664336","2022-01-14 16:33:35.664336","2022-01-14 18:14:17.622378"],"y":[31,31,16,16,1,1,25,25],"type":"scatter"},{"legendgroup":"rgb(93, 30, 135)","marker":{"color":"rgb(93, 30, 135)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null,null,null],"x":["2022-01-03 03:53:46.573427","2022-01-03 05:17:41.538462","2022-01-11 17:04:15.944056","2022-01-11 17:46:13.426573","2022-01-18 06:45:13.006993","2022-01-18 07:27:10.489510","2022-01-18 07:33:10.489510","2022-01-18 11:44:55.384615"],"y":[27,27,31,31,21,21,32,32],"type":"scatter"},{"legendgroup":"rgb(96, 18, 240)","marker":{"color":"rgb(96, 18, 240)","opacity":0,"size":1},"mode":"markers","name":"","showlegend":false,"text":[null,null,null,null,null,null],"x":["2022-01-08 08:53:28.951049","2022-01-17 02:40:53.706294","2022-01-17 02:46:53.706294","2022-01-18 06:45:13.006993","2022-01-18 06:51:13.006993","2022-02-01 06:31:04.615385"],"y":[9,9,21,21,15,15],"type":"scatter"}],                        {"height":600,"hovermode":"closest","showlegend":true,"title":{"text":"Job shop Schedule"},"xaxis":{"rangeselector":{"buttons":[{"count":7,"label":"1w","step":"day","stepmode":"backward"},{"count":1,"label":"1m","step":"month","stepmode":"backward"},{"count":6,"label":"6m","step":"month","stepmode":"backward"},{"count":1,"label":"YTD","step":"year","stepmode":"todate"},{"count":1,"label":"1y","step":"year","stepmode":"backward"},{"step":"all"}]},"showgrid":true,"type":"date","zeroline":false},"yaxis":{"autorange":false,"range":[-1,34],"showgrid":false,"ticktext":["Machine 6_7","Machine 4_5","Machine 6_6","Machine 4_4","Machine 4_3","Machine 6_5","Machine 6_4","Machine 4_2","Machine 7_7","Machine 7_6","Machine 6_3","Machine 6_2","Machine 6_1","Machine 4_1","Machine 7_5","Machine 7_4","Machine 3_2","Machine 1_8","Machine 1_7","Machine 1_6","Machine 1_5","Machine 3_1","Machine 7_3","Machine 1_4","Machine 1_3","Machine 1_2","Machine 2_3","Machine 1_1","Machine 2_2","Machine 7_2","Machine 7_1","Machine 2_1","Machine 5_1"],"tickvals":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],"zeroline":false},"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('b247bc65-91aa-46e2-91aa-0ef0b4efec79');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>