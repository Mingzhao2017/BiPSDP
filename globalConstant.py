# -*- coding:utf-8 -*-
# @Time: 2019/10/29  11:35
# @User: Barry W
# @Author: Mingzhao Wang (wangmz2014@163.com)
# @File: globalConstant.py :
# @Ref: 
import itertools

single_RNA = ['A', 'C', 'G', 'U']
single_RNA_withN = ['A', 'C', 'G', 'U', 'N']
di_RNA = []
tri_RNA = []
tetra_RNA = []
penta_RNA = []
six_RNA = []
seven_RNA = []
eight_RNA = []
nine_RNA = []
for diStrucPair in itertools.product(single_RNA, repeat=2):
    di_RNA.append(''.join(diStrucPair))
for triStrucPair in itertools.product(single_RNA, repeat=3):
    tri_RNA.append(''.join(triStrucPair))
for tetra in itertools.product(single_RNA, repeat=4):
    tetra_RNA.append(''.join(tetra))
for pentaStr in itertools.product(single_RNA, repeat=5):
    penta_RNA.append(''.join(pentaStr))
for sixStr in itertools.product(single_RNA, repeat=6):
    six_RNA.append(''.join(sixStr))
for sevenStr in itertools.product(single_RNA, repeat=7):
    seven_RNA.append(''.join(sevenStr))
for eightStr in itertools.product(single_RNA, repeat=8):
    eight_RNA.append(''.join(eightStr))
for nineStr in itertools.product(single_RNA, repeat=9):
    nine_RNA.append(''.join(nineStr))

##
single_DNA = ['A', 'C', 'G', 'T']
single_DNA_withN = ['A', 'C', 'G', 'T', 'N']
di_DNA = []
tri_DNA = []
tetra_DNA = []
penta_DNA = []
six_DNA = []
seven_DNA = []
eight_DNA = []
nine_DNA = []
for diStrucPair in itertools.product(single_DNA, repeat=2):
    di_DNA.append(''.join(diStrucPair))
for triStrucPair in itertools.product(single_DNA, repeat=3):
    tri_DNA.append(''.join(triStrucPair))
for tetra in itertools.product(single_DNA, repeat=4):
    tetra_DNA.append(''.join(tetra))
for pentaStr in itertools.product(single_DNA, repeat=5):
    penta_DNA.append(''.join(pentaStr))
for sixStr in itertools.product(single_DNA, repeat=6):
    six_DNA.append(''.join(sixStr))
for sevenStr in itertools.product(single_DNA, repeat=7):
    seven_DNA.append(''.join(sevenStr))
for eightStr in itertools.product(single_DNA, repeat=8):
    eight_DNA.append(''.join(eightStr))
for nineStr in itertools.product(single_DNA, repeat=9):
    nine_DNA.append(''.join(nineStr))

##
single_DNA2 = ['A', 'C', 'G', 'T', 'N']
di_DNA2 = []
tri_DNA2 = []
tetra_DNA2 = []
penta_DNA2 = []

for diStrucPair in itertools.product(single_DNA, repeat=2):
    di_DNA2.append(''.join(diStrucPair))
for triStrucPair in itertools.product(single_DNA, repeat=3):
    tri_DNA2.append(''.join(triStrucPair))
for tetra in itertools.product(single_DNA, repeat=4):
    tetra_DNA2.append(''.join(tetra))
for pentaStr in itertools.product(single_DNA, repeat=5):
    penta_DNA2.append(''.join(pentaStr))

##
dictPBERNA = {single_RNA[0]: [0, 0, 0, 1], single_RNA[1]: [0, 0, 1, 0],
              single_RNA[2]: [0, 1, 0, 0], single_RNA[3]: [1, 0, 0, 0]}
dictPBEDNA = {single_DNA[0]: [0, 0, 0, 1], single_DNA[1]: [0, 0, 1, 0],
              single_DNA[2]: [0, 1, 0, 0], single_DNA[3]: [1, 0, 0, 0]}

##
dictDPBERNA = {di_RNA[0]: [0, 0, 0, 0], di_RNA[1]: [0, 0, 0, 1], di_RNA[2]: [0, 0, 1, 0],
               di_RNA[3]: [0, 0, 1, 1], di_RNA[4]: [0, 1, 0, 0], di_RNA[5]: [0, 1, 0, 1],
               di_RNA[6]: [0, 1, 1, 0], di_RNA[7]: [0, 1, 1, 1], di_RNA[8]: [1, 0, 0, 0],
               di_RNA[9]: [1, 0, 0, 1], di_RNA[10]: [1, 0, 1, 0], di_RNA[11]: [1, 0, 1, 1],
               di_RNA[12]: [1, 1, 0, 0], di_RNA[13]: [1, 1, 0, 1],
               di_RNA[14]: [1, 1, 1, 0], di_RNA[15]: [1, 1, 1, 1]}
dictDPBEDNA = {di_DNA[0]: [0, 0, 0, 0], di_DNA[1]: [0, 0, 0, 1], di_DNA[2]: [0, 0, 1, 0],
               di_DNA[3]: [0, 0, 1, 1], di_DNA[4]: [0, 1, 0, 0], di_DNA[5]: [0, 1, 0, 1],
               di_DNA[6]: [0, 1, 1, 0], di_DNA[7]: [0, 1, 1, 1], di_DNA[8]: [1, 0, 0, 0],
               di_DNA[9]: [1, 0, 0, 1], di_DNA[10]: [1, 0, 1, 0], di_DNA[11]: [1, 0, 1, 1],
               di_DNA[12]: [1, 1, 0, 0], di_DNA[13]: [1, 1, 0, 1],
               di_DNA[14]: [1, 1, 1, 0], di_DNA[15]: [1, 1, 1, 1]}
##
structure = ['(((', '((.', '(..', '(.(', '.((', '.(.', '..(', '...']
strucList = []
for item in itertools.product(single_RNA, structure):
    str1 = item[0]
    str2 = item[1]
    strucList.append(str1+str2)

strucPairList = ['A', 'C', 'G', 'U', 'A-U', 'U-A', 'G-C', 'C-G', 'G-U', 'U-G']
di_strucPairList = []
tri_strucPairList = []
for diStrucPair in itertools.product(strucPairList, repeat=2):
    di_strucPairList.append(''.join(diStrucPair))
for triStrucPair in itertools.product(strucPairList, repeat=3):
    tri_strucPairList.append(''.join(triStrucPair))

dictFE = {strucPairList[0]: 0, strucPairList[1]: 0,
          strucPairList[2]: 0, strucPairList[3]: 0,
          strucPairList[4]: -2, strucPairList[5]: -2,
          strucPairList[6]: -3, strucPairList[7]: -3,
          strucPairList[8]: -1, strucPairList[9]: -1}