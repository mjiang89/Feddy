"""
Author: Meng Jiang (mjiang2@nd.edu)
"""
import os


def cleaning():
    header = 'TIME NODE TYPE POSITION_X POSITION_Y WIDTH HEIGHT CENTER_R CENTER_G CENTER_B ' \
            + 'TOP_R TOP_G TOP_B BOTTOM_R BOTTOM_G BOTTOM_B LEFT_R LEFT_G LEFT_B RIGHT_R RIGHT_G RIGHT_B'

    for [scene,numvideo] in [['bookstore',7],['coupa',4],['deathcircle',5],['gates',9], \
            ['hyang',15],['little',4],['nexus',12],['quad',4]]:

        for videoid in range(numvideo):

            tm2nodes = {}

            fr = open('raw/'+scene+'_video'+str(videoid)+'.attributes','r')
            fr.readline()
            for line in fr:
                arr = line.strip('\r\n').split(' ')
                isnan = False
                for i in range(7):
                    if arr[i] == 'nan':
                        isnan = True
                if isnan: continue
                tm = int(arr[0])
                node = int(arr[1])
                posx,posy = float(arr[2]),float(arr[3])
                width,height = float(arr[4]),float(arr[5])
                tp = int(arr[6])

                centerR,centerG,centerB = float(arr[9]),float(arr[8]),float(arr[7])
                if arr[9] == 'nan':
                    nonnan = False
                    for i in [12,15,18,21]:
                        if not arr[i] == 'nan':
                            centerR = float(arr[i])
                            nonnan = True
                            break
                    if not nonnan: continue
                if arr[8] == 'nan':
                    nonnan = False
                    for i in [11,14,17,20]:
                        if not arr[i] == 'nan':
                            centerG = float(arr[i])
                            nonnan = True
                            break
                    if not nonnan: continue
                if arr[7] == 'nan':
                    nonnan = False                    
                    for i in [10,13,16,19]:
                        if not arr[i] == 'nan':
                            centerG = float(arr[i])
                            nonnan = True
                            break
                    if not nonnan: continue

                topR,topG,topB = float(arr[12]),float(arr[11]),float(arr[10])
                bottomR,bottomG,bottomB = float(arr[15]),float(arr[14]),float(arr[13])
                rightR,rightG,rightB = float(arr[18]),float(arr[17]),float(arr[16])
                leftR,leftG,leftB = float(arr[21]),float(arr[20]),float(arr[19])

                if arr[12] == 'nan': topR = centerR
                if arr[11] == 'nan': topG = centerG
                if arr[10] == 'nan': topB = centerB
                
                if arr[15] == 'nan': bottomR = centerR
                if arr[14] == 'nan': bottomG = centerG
                if arr[13] == 'nan': bottomB = centerB

                if arr[18] == 'nan': rightR = centerR
                if arr[17] == 'nan': rightG = centerG
                if arr[16] == 'nan': rightB = centerB

                if arr[21] == 'nan': leftR = centerR
                if arr[20] == 'nan': leftG = centerG
                if arr[19] == 'nan': leftB = centerB

                if not tm in tm2nodes:
                    tm2nodes[tm] = []

                tm2nodes[tm].append([node,tp,posx,posy,width,height, \
                        centerR,centerG,centerB, \
                        topR,topG,topB, \
                        bottomR,bottomG,bottomB, \
                        leftR,leftG,leftB, \
                        rightR,rightG,rightB])
            fr.close()

            fw = open('data/camp'+scene[0]+str(videoid)+'.txt','w')
            fw.write(header+'\n')
            for [tm,nodes] in sorted(tm2nodes.items(),key=lambda x:x[0]):
                for node in sorted(nodes,key=lambda x:x[0]):
                    s = str(tm)+' '+str(node[0])+' '+str(node[1])
                    for i in range(2,len(node)):
                        s += ' '+("%.2f" % node[i])
                    fw.write(s+'\n')
            fw.close()

def stats():
    dirname = 'data'
    extname = '.txt'
 
    filenames = []
    for filename in os.listdir(dirname):
        if filename.endswith(extname):
            filenames.append(filename)
    filenames = sorted(filenames)

    fw = open('data-stats.txt','w')
    fw.write('SCENE\tNUM_TIME\tMIN_TIME\tMAX_TIME\tNUM_NODE\tMIN_NODE\tMAX_NODE\tTYPE_COUNT\n')
    for filename in filenames:
        filepath = os.path.join('data',filename)
        timeset,node2tp,tp2count = set(),{},{}
        fr = open(filepath,'r')
        fr.readline()
        for line in fr:
            arr = line.strip('\r\n').split(' ')
            tm,node,tp = int(arr[0]),int(arr[1]),int(arr[2])
            timeset.add(tm)
            node2tp[node] = tp
        fr.close()
        if len(timeset) == 0: continue
        numtm = len(timeset)
        mintm = min(timeset)
        maxtm = max(timeset)
        numnode = len(node2tp.keys())
        minnode = min(node2tp.keys())
        maxnode = max(node2tp.keys())
        for [node,tp] in node2tp.items():
            if not tp in tp2count:
                tp2count[tp] = 0
            tp2count[tp] += 1
        s = ''
        for [tp,count] in sorted(tp2count.items(),key=lambda x:x[0]):
            s += '|'+str(tp)+':'+str(count)
        fw.write(filename[:-len(extname)]+'\t'+str(numtm)+'\t'+str(mintm)+'\t'+str(maxtm) \
                +'\t'+str(numnode)+'\t'+str(minnode)+'\t'+str(maxnode)+'\t'+s[1:]+'\n')
    fw.close()

if __name__ == '__main__':
    cleaning()
    stats()
