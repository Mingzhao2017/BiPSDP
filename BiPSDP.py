# -*- coding:utf-8 -*-
# @User: Barry W
# @Author: Mingzhao Wang 
# @File: 
# @Ref:

import csv
import time, sys, math, os
import numpy as np
import globalConstant

def make_PSPPMI_vector(ksai, seq_list, label, len_seq, nucleType):
    '''
        :param k: the number of nucleotide
        :param ksai: ksai is the interval of the two nucleotides in a pair
        :param file_path: the path of data
        :return: NPPS vector
        '''
    if ksai > math.floor((len_seq - 3) / 2):
        print('ERROR: The value of ksai must <= %d-2, and the code will stop!' % math.floor((len_seq - 3) / 2))
        sys.exit(0)
    np_seq_list = np.array(seq_list)
    posi_data, nega_data = split_data_posi_nega(np_seq_list, label)
    # ksai = len_seq - k
    flag = True
    for j in range(ksai + 1):
        print('ksai:', j)
        posi_di_fre_forward_matrix = calculate_frequency(posi_data, len_seq, j, nucleType, direction='forward')
        posi_di_fre_reverse_matrix = calculate_frequency(posi_data, len_seq, j, nucleType, direction='reverse')
        nega_di_fre_forward_matrix = calculate_frequency(nega_data, len_seq, j, nucleType, direction='forward')
        nega_di_fre_reverse_matrix = calculate_frequency(nega_data, len_seq, j, nucleType, direction='reverse')
        ksai_vector = calculate_vector(np_seq_list, posi_di_fre_forward_matrix, posi_di_fre_reverse_matrix,
                                       nega_di_fre_forward_matrix, nega_di_fre_reverse_matrix, len_seq, j, nucleType)
        if flag:
            vector_PSPNMI = ksai_vector
            flag = False
        else:
            vector_PSPNMI = np.hstack((vector_PSPNMI, ksai_vector))
    return vector_PSPNMI

def calculate_vector(data,  posi_di_fre_forward_matrix, posi_di_fre_reverse_matrix,
                     nega_di_fre_forward_matrix, nega_di_fre_reverse_matrix,
                     len_seq, ksai, nucleType):
    '''
    :param data: positive or negative data
    :param posi_single_fre_matrix: frequence matrix of single nucleotide in positive data
    :param nega_single_fre_matrix: frequence matrix of single nucleotide in negative data
    :param posi_di_fre_forward_matrix: forward direction frequence matrix of dinucleotide in positive data
    :param posi_di_fre_reverse_matrix: reverse direction frequence matrix of dinucleotide in positive data
    :param nega_di_fre_forward_matrix: forward direction frequence matrix of dinucleotide in negative data
    :param nega_di_fre_reverse_matrix: reverse direction frequence matrix of dinucleotide in negative data
    :param len_seq: the length of sequence
    :param k: k-nucleotide
    :param ksai: ksai is the interval of the two nucleotides in a pair
    :return: frequence matrix
    '''
    vector = np.zeros((len(data), len_seq - 2 * ksai - 2))
    for seqIndex in range(len(data)):
        temp_vector = []
        seq = data[seqIndex]
        for i in range(ksai + 1, len_seq - 2 - ksai + 1):
            di_nucle_forward = seq[i] + seq[i + ksai + 1]  # 当前序列的第i个和第i+ksai+1个核苷酸组成的二核苷酸
            di_nucle_reverse = seq[i] + seq[i - ksai - 1]  # 当前序列的第i个和第i-ksai-1个核苷酸组成的二核苷酸
            if nucleType == 'RNA':
                di_forward_index = globalConstant.di_RNA.index(di_nucle_forward)
                di_reverse_index = globalConstant.di_RNA.index(di_nucle_reverse)
            elif nucleType == 'DNA':
                di_forward_index = globalConstant.di_DNA.index(di_nucle_forward)
                di_reverse_index = globalConstant.di_DNA.index(di_nucle_reverse)
            di_posi_seq_vector_forward = posi_di_fre_forward_matrix[:, i]  # 正类前向二核苷酸概率矩阵的第i列
            di_posi_seq_vector_reverse = posi_di_fre_reverse_matrix[:, i - ksai - 1]  # 正类后向二核苷酸概率矩阵的第i列
            di_nega_seq_vector_forward = nega_di_fre_forward_matrix[:, i]  # 负类前向二核苷酸概率矩阵的第i列
            di_nega_seq_vector_reverse = nega_di_fre_reverse_matrix[:, i - ksai - 1]  # 负类后向二核苷酸概率矩阵的第i列
            di_value_posi_forward = di_posi_seq_vector_forward[di_forward_index]
            di_value_posi_reverse = di_posi_seq_vector_reverse[di_reverse_index]
            di_value_nega_forward = di_nega_seq_vector_forward[di_forward_index]
            di_value_nega_reverse = di_nega_seq_vector_reverse[di_reverse_index]
            di_posi_seq_vector = (di_value_posi_forward + di_value_posi_reverse) / 2
            di_nega_seq_vector = (di_value_nega_forward + di_value_nega_reverse) / 2
            value_fre = di_posi_seq_vector - di_nega_seq_vector
            temp_vector.append(value_fre)
        vector[seqIndex, :] = temp_vector
    return vector

def read_data(file_path, nucleType):
    '''
    :param file_path: the path of data
    :return: data with list type and label
    '''
    seq_list = []
    label = []
    if nucleType == 'RNA':
        startTuple = tuple(globalConstant.single_RNA)
    elif nucleType == 'DNA':
        startTuple = tuple(globalConstant.single_DNA)
    with open(file_path) as files:
        for line in files:
            if line.startswith(startTuple):
                line = line.rstrip('\n').strip()
                len_line = len(line)
                seq_list.append(line)
            else:
                seq_name = line
                P_or_N = seq_name[1]
                if P_or_N == 'P' or P_or_N == '+':
                    label.append(1)
                elif P_or_N == 'N' or P_or_N == '-':
                    label.append(2)
    return seq_list, label, len_line

def split_data_posi_nega(data, label):
    '''
    :param data: data with positive and negative
    :return: positive and negative data
    '''
    posi_index = [i for i, x in enumerate(label) if x == 1]
    nega_index = [i for i, x in enumerate(label) if x == 2]
    posi_data = data[posi_index]
    nega_data = data[nega_index]
    return posi_data, nega_data

def calculate_frequency(data, len_seq, ksai, nucleType, direction):
    '''
    :param data: positive or negative data
    :param k: the number of nucleotide, k-nucleotide
    :param len_seq: the length of sequence
    :param ksai: the interval of between first nucleotides and second nucleotides [0,l-k]
    :param direction: the direction of finding the next nucleotides, forward or reverse
    :return: frequence vector of sequence
    '''
    if nucleType == 'RNA':
        fre = np.zeros((globalConstant.di_RNA.__len__(), len_seq - 2 - ksai + 1))
    elif nucleType == 'DNA':
        fre = np.zeros((globalConstant.di_DNA.__len__(), len_seq - 2 - ksai + 1))
    if direction == 'forward':
        column_index = 0
        for len in range(len_seq - 2 - ksai + 1):
            row_seq = []
            temp_fre = []
            for line in data:
                row_seq.append(line[len] + line[len + ksai + 1])
            row_seq = np.array(row_seq)
            if nucleType == 'RNA':
                for nucle in globalConstant.di_RNA:
                    num_nucle = row_seq[row_seq == nucle]
                    temp_fre.append(float(num_nucle.size) / data.size)
            elif nucleType == 'DNA':
                for nucle in globalConstant.di_DNA:
                    num_nucle = row_seq[row_seq == nucle]
                    temp_fre.append(float(num_nucle.size) / data.size)
            fre[:, column_index] = temp_fre
            column_index += 1
    elif direction == 'reverse':
        column_index = 0
        for len in range(ksai + 1, len_seq):
            row_seq = []
            temp_fre = []
            for line in data:
                row_seq.append(line[len] + line[len - ksai - 1])
            row_seq = np.array(row_seq)
            if nucleType == 'RNA':
                for nucle in globalConstant.di_RNA:
                    num_nucle = row_seq[row_seq == nucle]
                    temp_fre.append(float(num_nucle.size) / data.size)
            elif nucleType == 'DNA':
                for nucle in globalConstant.di_DNA:
                    num_nucle = row_seq[row_seq == nucle]
                    temp_fre.append(float(num_nucle.size) / data.size)
            fre[:, column_index] = temp_fre
            column_index += 1
    return fre

def save_result(filePath, nucleType):
    '''
    :param filePath: the path of data
    :return: none
    '''
    startTime = time.time()
    k = 2  #:param k: the number of nucleotide
    splitPath = filePath.split('/')
    dataName = splitPath[3].split('.')[0]
    seq_list, label, len_seq = read_data(filePath, nucleType)
    # for times in range(math.floor((len_seq - 3) / 2) + 1):  # Arabidopsis（12），其余25
    ksai = math.floor((len_seq - 3) / 2)
    start_time = time.time()
    vectorPSPNMI = make_PSPPMI_vector(ksai, seq_list, label, len_seq, nucleType)
    print('data:', dataName)
    print('Done.')
    print('Used time: %.2fs' % (time.time() - start_time))
    np.savetxt("./result/data_" + dataName + "_BiPSDP_ksai_" + str(ksai) + ".csv", vectorPSPNMI, delimiter=" ")
    arrayLabel = np.array(label)
    np.savetxt("./result/label_" + dataName + "_BiPSDP.csv", arrayLabel, fmt='%d', delimiter=" ")

if __name__ == '__main__':
    # nucleType = 'DNA'
    nucleType = 'RNA'
    ################################################################
    if nucleType == 'DNA':
        fPath = './data/DNA-Running'
    elif nucleType == 'RNA':
        fPath = './data/RNA-Running'
    for i, j, k in os.walk(fPath):
        for fName in k:
            print(fName)
            file_path = fPath + '/' + fName
            save_result(file_path, nucleType)
    print('Done all.')



