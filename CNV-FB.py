import pandas as pd
import numpy as np
from numba import njit

from sklearn.metrics import euclidean_distances

from preprocess import preProcessData

def dis_matrix(RD_count):
    # calculating euclidean_distances matrix
    RD_count = RD_count.astype(np.float)
    pos = np.array(range(1, len(RD_count)+1))
    nr_min = np.min(RD_count)
    nr_max = np.max(RD_count)
    newpos = (pos - min(pos)) / (max(pos) - min(pos)) * (nr_max - nr_min) + nr_min

    RD_count = RD_count.astype(np.float)
    newpos = newpos.astype(np.float)
    rd = np.c_[newpos, RD_count]
    dis = euclidean_distances(rd, rd)
    return dis, newpos

@njit
def k_matrix(dis, k):
    min_matrix = np.zeros((dis.shape[0], k))
    for i in range(dis.shape[0]):
        sort = np.argsort(dis[i])
        min_row = dis[i][sort[k + 1]]
        for j in range(1, k + 1):
            min_matrix[i, j] = sort[j]
        dis[i][sort[1:(k + 1)]] = min_row
    return dis, min_matrix


@njit
def reach_density(dis, min_matrix, k):
    density = []
    for i in range(min_matrix.shape[0]):
        cur_sum = np.sum(dis[min_matrix[i], i])
        if cur_sum == 0.0:
            cur_density = 100
        else:
            cur_density = 1 / (cur_sum / k)
        density.append(cur_density)
    return density


def get_scores(density, min_matrix, binHead, k):
    scores = np.full(int(len(binHead)), 0.0)
    for i in range(min_matrix.shape[0]):
        cur_rito = density[min_matrix[i]] / density[i]
        cur_sum = np.sum(cur_rito) / k
        scores[i] = cur_sum
    return scores

def mad(scores):
    median = np.median(scores)
    b = 1.4826# 这个值应该是看需求加的，有点类似加大波动范围之类的
    mad = b * np.median(np.abs(scores - median))
    lower_limit = median - (3* mad)
    upper_limit = median + (3* mad)
    return upper_limit
    ground_truth = pd.read_table(groudtruth_path)
    truth_type = ground_truth["variant type"].tolist()
    truth_start = ground_truth['start'].tolist()
    truth_end = ground_truth['stop'].tolist()

    count = 0
    for i in range(len(result_type)):
        for j in range(len(truth_type)):
            if truth_start[j] <= result_start[i] <= truth_end[j] and truth_type[j] == result_type[i]:
                if result_end[i] <= truth_end[j]:
                    count += (result_end[i] - result_start[i] + 1)
                    print(f"{result_start[i]}   {result_end[i]}")
                elif result_end[i] > truth_end[j]:
                    count += (truth_end[j] - result_start[i] + 1)
                    print(f"{result_start[i]}   {result_end[i]}")

            elif truth_start[j] >= result_start[i] and truth_type[j] == result_type[i]:
                if truth_start[j] <= result_end[i] <= truth_end[j]:
                    count += (result_end[i] - truth_start[j] + 1)
                    print(f"{result_start[i]}   {result_end[i]}")

                elif result_end[i] >= truth_end[j]:
                    count += (truth_end[j] - truth_start[j] + 1)
                    print(f"{result_start[i]}   {result_end[i]}")

    result_count = 0
    for i in range(len(result_start)):
        result_count += (result_end[i] - result_start[i] + 1)

    truth_count = 0
    for i in range(len(truth_start)):
        truth_count += (truth_end[i] - truth_start[i] + 1)

    if result_count == 0:
        precision = 0
    else:
        precision = count / result_count
    sensitivity = count / truth_count
    print("ans =", precision, sensitivity)

    return [precision, sensitivity]

    # ground_truth = pd.read_csv(groudtruth_path)
    # truth_start = ground_truth['binStart'].tolist()
    # truth_end = ground_truth['binEnd'].tolist()
    ground_truth=pd.read_table(groudtruth_path)
    truth_start = ground_truth['start'].tolist()
    truth_end = ground_truth['stop'].tolist()

    count = 0
    for i in range(len(result_start)):
        for j in range(len(truth_start)):
            if truth_start[j] <= result_start[i] <= truth_end[j]:
                if result_end[i] <= truth_end[j]:
                    count += (result_end[i] - result_start[i] + 1)
                elif result_end[i] > truth_end[j]:
                    count += (truth_end[j] - result_start[i] + 1)


            elif truth_start[j] >= result_start[i]:
                if truth_start[j] <= result_end[i] <= truth_end[j]:
                    count += (result_end[i] - truth_start[j] + 1)
                elif result_end[i] >= truth_end[j]:
                    count += (truth_end[j] - truth_start[j] + 1)


    result_count = 0
    for i in range(len(result_start)):
        result_count += (result_end[i] - result_start[i] + 1)

    truth_count = 440000
    if result_count == 0:
        precision = 0
    else:
        precision = count / result_count
    sensitivity = count / truth_count
    return [precision, sensitivity]

def lof(all_RD,k):
    print("calculating scores...")
    dis, newpos = dis_matrix(all_RD)
    dis, min_matrix = k_matrix(dis, k)
    min_matrix = min_matrix.astype(np.int)
    density = reach_density(dis, min_matrix, k)
    density = np.array(density)
    scores = get_scores(density, min_matrix, all_RD, k)
    return scores


def run(bamFilePath,refPath,n_est,segNumber):
    k=10
    all_chr,all_start,all_end,all_RD,mode= preProcessData(refPath, bamFilePath,segNumber)
    trainData = []
    for index in range(len(all_RD)):
        trainData.append([index+1,all_RD[index]])
    scoresVector=[]
    listIndex=np.arange(0,len(all_RD))
    indexRecord=[]

    np.random.seed(2022)
    for i in range(n_est):
        chooseIndex=np.random.choice(listIndex,np.random.randint(int(len(all_RD)/2),len(all_RD)))
        indexRecord.append(chooseIndex)
        tempTrainData=[]
        for i in chooseIndex:
            tempTrainData.append(all_RD[i])
        tempTrainData=np.array(tempTrainData)
        scores=lof(tempTrainData,k)
        temp=[]
        temp.append(scores)
        temp.append(chooseIndex)
        scoresVector.append(temp)

    scoresVector=np.array(scoresVector)
    finalScores=np.zeros(len(all_RD))
    indexCount=np.zeros(len(all_RD))

    #combining
    #Cumulative Sum
    for i in range(n_est):
        index=scoresVector[i][1]
        scores=scoresVector[i][0]
        for j in range(len(index)):
            finalScores[index[j]]+=scores[j]
            indexCount[index[j]]+=1

    print(finalScores)
    upper=mad(finalScores)
    index=finalScores>upper
    start=all_start[index]
    end=all_end[index]
    rd=all_RD[index]
    cnvRegion=[]

    result_start=[]
    result_end=[]
    result_type=[]
    for i in range(len(start)):
        state='duplication'
        if rd[i]<mode:
            state='deletion'
        result_start.append(start[i])
        result_end.append(end[i])
        result_type.append(state)
        cnvRegion.append([start[i],end[i],state])
    df=pd.DataFrame(cnvRegion,columns=['binStart','binEnd','state'])
    fileName=bamFilePath.split('/')[-1]
    df.to_csv(f'./{fileName}.csv')


if __name__=='__main__':

    bamFilePath = "./realData/NA12878.chrom21.SLX.maq.SRP000032.2009_07.bam"
    refPath='./realData/'

    run(bamFilePath, refPath, 70, '1')
