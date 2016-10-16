def get_ctgy(fname):#根据文件名称获取类别的数字编号
        index = {'fi':0,'lo':1,'co':2,'ho':3,'ed':4,'te':5,
                 'ca':6,'ta':7,'sp':8,'he':9,'ar':10,'fu':11}
        return index[fname[:2]]

def updateWeight():

        #EP_post是 单词数*类别 的矩阵
        for i in range(wordNum):
            for j in range(ctgyNum):
                EP_post[i][j] = 0.0 #[[0.0 for x in range(ctgyNum)] for y in range(wordNum)]
        # prob是 文本数*类别 的矩阵，记录每个文本属于每个类别的概率
        cond_prob_textNum_ctgyNum = [[0.0 for x in range(ctgyNum)] for y in range(textNum)]
        #计算p(类别|文本)

        for i in range(textNum):#对每一个文本
                zw = 0.0  #归一化因子
                for j in range(ctgyNum):#对每一个类别
                        tmp = 0.0
                        #texts_list_dict每个元素对应一个文本，该元素的元素是单词序号：频率所组成的字典。
                        for (feature,feature_value) in texts_list_dict[i].items():
                            #v就是特征f(x,y)，非二值函数，而是实数函数，
                            #k是某文本中的单词，v是该单词的次数除以该文本不重复单词的个数。
                                #feature_weight是 单词数*类别 的矩阵，与EP_prior相同
                                tmp+=feature_weight[feature][j]*feature_value #feature_weight是最终要求的模型参数，其元素与特征一一对应，即一个特征对应一个权值
                        tmp = math.exp(tmp)
                        zw+=tmp #zw关于类别求和
                        cond_prob_textNum_ctgyNum[i][j]=tmp                        
                for j in range(ctgyNum):
                        cond_prob_textNum_ctgyNum[i][j]/=zw
        #上面的部分根据当前的feature_weight矩阵计算得到prob矩阵（文本数*类别的矩阵，每个元素表示文本属于某类别的概率），
        #下面的部分根据prob矩阵更新feature_weight矩阵。


        for x in range(textNum):
                ctgy = category[x] #根据文本序号获取类别序号
                for (feature,feature_value) in texts_list_dict[x].items(): #该文本中的单词和对应的频率
                        EP_post[feature][ctgy] += (cond_prob_textNum_ctgyNum[x][ctgy]*feature_value)#认p(x)的先验概率相同        
        #更新特征函数的权重w
        for i in range(wordNum):
                for j in range(ctgyNum):
                        if (EP_post[i][j]<1e-17) |  (EP_prior[i][j]<1e-17) :
                                continue                        

                        feature_weight[i][j] += math.log(EP_prior[i][j]/EP_post[i][j])        

def modelTest():
        testFiles = os.listdir('data\\test\\')
        errorCnt = 0
        totalCnt = 0

        #matrix是类别数*类别数的矩阵，存储评判结果
        matrix = [[0 for x in range(ctgyNum)] for y in range(ctgyNum)]
        for fname in testFiles: #对每个文件

                lines = open('data\\test\\'+fname)
                ctgy = get_ctgy(fname) #根据文件名的前两个字符给出类别的序号
                probEst = [0.0 for x in range(ctgyNum)]         #各类别的后验概率
                for line in lines: #该文件的每一行是一个单词和该单词在该文件中出现的频率
                        arr = line.split('\t')
                        if not words_dict.has_key(arr[0]) : 
                            continue        #测试集中的单词如果在训练集中没有出现则直接忽略
                        word_id,freq = words_dict[arr[0]],float(arr[1])
                        for index in range(ctgyNum):#对于每个类别
                            #feature_weight是单词数*类别墅的矩阵
                            probEst[index] += feature_weight[word_id][index]*freq
                ctgyEst = 0
                maxProb = -1
                for index in range(ctgyNum):
                        if probEst[index]>maxProb:
                            ctgyEst = index
                            maxProb = probEst[index]
                totalCnt+=1
                if ctgyEst!=ctgy: 
                    errorCnt+=1
                matrix[ctgy][ctgyEst]+=1
                lines.close()
        #print "%-5s" % ("类别"),
        #for i in range(ctgyNum):
        #    print "%-5s" % (ctgyName[i]),  
        #print '\n',
        #for i in range(ctgyNum):
        #    print "%-5s" % (ctgyName[i]), 
        #    for j in range(ctgyNum):
        #        print "%-5d" % (matrix[i][j]), 
        #    print '\n',
        print "测试总文本个数:"+str(totalCnt)+"  总错误个数:"+str(errorCnt)+"  总错误率:"+str(errorCnt/float(totalCnt))

def prepare():
        i = 0
        lines = open('data\\words.txt').readlines()
        #words_dict给出了每一个中文词及其对应的全局统一的序号，是字典类型，示例：{'\xd0\xde\xb5\xc0\xd4\xba': 0}
        for word in lines:
                word = word.strip()
                words_dict[word] = i
                i+=1
        #计算约束函数f的经验期望EP(f)
        files = os.listdir('data\\train\\') #train下面都是.txt文件
        index = 0
        for fname in files: #对训练数据集中的每个文本文件
                file_feature_dict = {}
                lines = open('data\\train\\'+fname)
                ctgy = get_ctgy(fname) #根据文件名的前两个汉字，也就是中文类别来确定类别的序号

                category[index] = ctgy #data/train/下每个文本对应的类别序号
                for line in lines: #每行内容：古迹  0.00980392156863
                        # line的第一个字符串是中文单词，第二个字符串是该单词的频率
                        arr = line.split('\t')
                        #获取单词的序号和频率
                        word_id,freq= words_dict[arr[0]],float(arr[1])

                        file_feature_dict[word_id] = freq
                        #EP_prior是单词数*类别的矩阵
                        EP_prior[word_id][ctgy]+=freq
                texts_list_dict[index] = file_feature_dict
                index+=1
                lines.close()
def train():
        for loop in range(4):
            print "迭代%d次后的模型效果：" % loop
            updateWeight()
            modelTest()


textNum = 2741  # data/train/下的文件的个数
wordNum = 44120 #data/words.txt的单词数，也是行数
ctgyNum = 12


#feature_weight是单词数*类别墅的矩阵
feature_weight = np.zeros((wordNum,ctgyNum))#[[0 for x in range(ctgyNum)] for y in range(wordNum)]

ctgyName = ['财经','地域','电脑','房产','教育','科技','汽车','人才','体育','卫生','艺术','娱乐']
words_dict = {}

# EP_prior是个12（类）* 44197（所有类的单词数）的矩阵，存储对应的频率
EP_prior = np.zeros((wordNum,ctgyNum))
EP_post = np.zeros((wordNum,ctgyNum))
#print np.shape(EP_prior)
texts_list_dict = [0]*textNum #所有的训练文本 
category = [0]*textNum        #每个文本对应的类别

print "初始化:......"
prepare()
print "初始化完毕，进行权重训练....."
train()
