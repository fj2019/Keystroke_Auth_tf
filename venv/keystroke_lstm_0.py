# encoding=utf-8
import numpy
import random

import os

#from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding,Bidirectional
from keras.layers import LSTM, GRU
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import load_model

class keystroke_lstm:
    seq_length = 30
    batch_size = 10
    num_inputs = 1000
    max_len = 5
    epochs = 500

    filter_length = 2  # 滤波器长度
    nb_filter = 32  # 滤波器个数
    pool_length = 2  # 池化长度

    usernum=76
    def split(self,str):
        num = []
        list = str.split(',')
        # print(list)
        for i in range(len(list)):
            if list[i].strip() == "" or list[i] == "*":
                num.append(0)
            else:
                num.append(float(list[i]))
        # print(num)
        return num

    def load_txt(self,path, file):
        keycode = {}
        for i in range(65, 91):
            keycode[chr(i)] = i
        keycode['LMenu'] = 1
        keycode['Tab'] = 2
        keycode['Return'] = 3
        keycode['LShiftKey'] = 4
        keycode['Space'] = 32
        keycode['Back'] = 5
        keycode['OemPeriod'] = 6
        keycode['Oemcomma'] = 7

        keycode['OemMinus'] = 9
        for j in range(10):
            str1 = "D" + str(j)
            keycode[str1] = j + 10
        keycode['RShiftKey'] = 20
        keycode['OemQuestion'] = 21
        keycode['Oemplus'] = 22
        for j in range(10):
            str1 = "Oem" + str(j)
            keycode[str1] = j + 100
        keycode['Capital'] = 23
        keycode['LControlKey'] = 24
        keycode['RControlKey'] = 25
        keycode['Up'] = 26
        keycode['Down'] = 27
        keycode['Left'] = 28
        keycode['Right'] = 29
        keycode['Escape'] = 30
        keycode['Delete'] = 31
        keycode['End'] = 33
        keycode['Insert'] = 34
        keycode['OemOpenBrackets'] = 35

        for j in range(10):
            str1 = "NumPad" + str(j)
            keycode[str1] = j + 36
        for j in range(1, 11):
            str1 = "F" + str(j)
            keycode[str1] = j + 46
        keycode['LButton,'] = 56
        keycode['Apps'] = 57
        keycode['RWin'] = 58
        keycode['RMenu'] = 59
        keycode['Oemtilde'] = 60
        keycode['Home'] = 61
        keycode['Next'] = 62
        keycode['PageUp'] = 63
        keycode['NumLock'] = 64
        keycode['Clear'] = 110
        keycode['Add'] = 111
        keycode['F12'] = 112
        keycode['Subtract'] = 113
        keycode['Decimal'] = 114
        keycode['LWin'] = 115
        keycode['PrintScreen'] = 116
        keycode['MediaPreviousTrack'] = 117
        keycode['Scroll'] = 118
        keycode['BrowserHome'] = 119
        keycode['BrowserForward'] = 120
        keycode['BrowserBack'] = 121
        keycode['Divide'] = 122
        keycode['VolumeMute'] = 123
        keycode['Pause'] = 123
        len1 = 0
        for line in open(path + file):
            len1 = len1 + 1
        # print(keycode)
        # print(len1)
        num = 0
        d = []
        # print(int(len1/2))
        keytimelist = [[] for i in range(int(len1 / 2) + 10)]
        print(path + file)
        for line in open(path + file):

            # if numint(len1/2)>=3100 and num<3300:
            # print(line)
            keylist = []
            if file[3] == '.':
                keylist = []
                tmp = line.split(' ')
                if len(tmp) == 6:
                    keylist.append(int(tmp[1]))
                    keylist.append(tmp[2].capitalize() + tmp[4].capitalize())
                    keylist.append(float(tmp[5]))
                else:
                    keylist.append(int(tmp[1]))
                    keylist.append(tmp[2].capitalize() + tmp[3].capitalize())
                    keylist.append(float(tmp[4]))
                print(keylist)
                print(num)
            else:
                keylist = line.split(' ')
                keylist[0] = keycode[keylist[0]]
                keylist[2] = float(keylist[2])
            if len(d) == 0:
                if keylist[1] == "KeyDown":
                    keylist.append(num)
                    d.append(keylist)
                    num = num + 1
            else:
                f = -1
                for i in range(len(d)):
                    if d[i][0] == keylist[0]:
                        f = i
                        break
                if f == -1:
                    if keylist[1] == "KeyDown":
                        keylist.append(num)
                        d.append(keylist)
                        num = num + 1
                else:

                    # if  num<1700:
                    # print(d[f])

                    if keylist[1] == "KeyUp":
                        '''
    					if num>=700 and num<800:
    						 print(d[f])
    					'''
                        # print(d[f][3])
                        keytimelist[d[f][3]].append(keylist[0])
                        keytimelist[d[f][3]].append(d[f][2])
                        keytimelist[d[f][3]].append(keylist[2])
                        del d[f]

        for i in range(len(keytimelist) - 1, -1, -1):
            if len(keytimelist[i]) == 0:
                keytimelist.pop(i)

        return keytimelist

    # load_txt("/home/zsf/python/free_txt/baseline_2/","034201.txt")
    def solve_data(self):
        X = []
        Y = []
        for filenum in range(3):
            # dirc = "rotation_" + str(filenum+1) + "/"
            # path = "free-text-dif/" + dirc  # 文件夹目录
            dirc = "baseline_" + str(filenum) + "/"
            path = "free_txt/" + dirc  # 文件夹目录

            files = os.listdir(path)
            for file in files:
                # print(file)
                xx = []
                yy = []

                keylist = self.load_txt(path, file)
                #print(keylist)
                key_word_list = []
                for i in range(len(keylist) - 1):
                    key_word = []
                    key_word.append(keylist[i][0])
                    key_word.append(keylist[i + 1][0])
                    # key_word.append(keylist[i+2][0])

                    key_word.append(keylist[i][2] - keylist[i][1])
                    key_word.append(keylist[i + 1][2] - keylist[i + 1][1])
                    # key_word.append(keylist[i+2][2]-keylist[i+2][1])

                    key_word.append(keylist[i + 1][1] - keylist[i][2])
                    key_word.append(keylist[i + 1][1] - keylist[i][1])

                    # key_word.append(keylist[i+2][1]-keylist[i][2])
                    # $key_word.append(keylist[i+2][1]-keylist[i][1])

                    # key_word.append(keylist[i+2][1]-keylist[i+1][2])
                    # key_word.append(keylist[i+2][1]-keylist[i+1][1])
                    # key_word.append(keylist[i+1][2]-keylist[i][1])
                    # key_word.append(keylist[i+1][2]-keylist[i][2])
                    key_word_list.append(key_word)

                l = 0
                r = self.seq_length
                # print(r)
                len1 = len(key_word_list)
                # print(len1)
                # print(key_word_list)
                for i in range(int(len1 / self.seq_length)):
                    xx.append(key_word_list[l:r])
                    yy.append(int(file[:3]) - 1)
                    l = l + self.seq_length
                    r = r + self.seq_length

                X = X + xx
                Y = Y + yy
        # print(X[0],Y[0])
        print(Y)
        print(self.usernum)
        keyer = [[] for i in range(self.usernum)]
        for i in range(len(Y)):
            #print(Y[i])
            keyer[Y[i]].append(X[i])
        # 数据归一化

        return keyer

    def load_txt_keyword(self,file):
        num = 0
        for line in open(file):
            # print (line)
            if line.strip() != "":
                num = num + 1
                if num == 1:
                    index = int(line)
                elif num == 2:
                    keydownindex = split(line)
                elif num == 3:
                    keydowntime = split(line)
                elif num == 4:
                    keyupindex = split(line)
                elif num == 5:
                    keyuptime = split(line)

        return index, keydownindex, keydowntime, keyupindex, keyuptime

    def split_list(self,n, list):
        num = int(len(list) / 10)
        l = num * n
        if n == 9:
            r = len(list)
        else:
            r = num * n + num
        # print(num,l,r)
        X_test = list[l:r]
        if l == 0:
            X_train = list[r:]
        elif r == len(list):
            X_train = list[:l]
        else:
            X_train = list[:l] + list[r:]
        return X_train, X_test

    def small_list(self,test):
        l = 0
        r = self.seq_length
        len1 = len(test)
        xx = []
        while r < len1:
            xx.append(test[l:r])
            l = l + 1
            r = r + 1
        if r > len1:
            xx.append(test[len1 - seq_length:len1])
        return xx

    def count_if(self,predict, len):
        count = [0] * 4
        for k in range(len):
            if predict[k] <= 0.5:
                count[0] = count[0] + 1
            if predict[k] <= 0.6:
                count[1] = count[1] + 1
            if predict[k] <= 0.7:
                count[2] = count[2] + 1
            if predict[k] <= 0.8:
                count[3] = count[3] + 1
        print("阈值50：%f,60：%f,70：%f,80：%f" % (count[0] / len, count[1] / len, count[2] / len, count[3] / len))

    def setThr(self,model, x_test, num):
        x_test = x_test.tolist()
        x_test_1 = x_test[:num]
        # print(len(x_test_1),x_test_1)
        x_test_0 = x_test[num:]
        test_1 = x_test_1[0]

        for i in range(1, num):
            # print(test_1,len(test_1))
            test_1 = test_1 + x_test_1[i]
        # print(len(test_1),test_1)
        test_0 = x_test_0[0]
        for j in range(1, len(x_test_0)):
            test_0 = test_0 + x_test_0[j]
        xx = small_list(test_1)
        # print(xx)
        len1 = len(xx)
        num = 10 - (len1 % 10)
        for i in range(num):
            xx.append(xx[0])
        xx = numpy.array(xx)
        predict_1 = model.predict(xx, batch_size=batch_size)
        print("主人：")
        count_if(predict_1, len1)

        xx = small_list(test_0)
        len0 = len(xx)
        num = 10 - (len0 % 10)
        for i in range(num):
            xx.append(xx[0])
        xx = numpy.array(xx)
        predict_0 = model.predict(xx, batch_size=batch_size)
        print("入侵者：")
        count_if(predict_0, len0)

    def lstm(self):

        for seq_len in [30]:

            print(self.seq_length)
            keyer = self.solve_data()

            toperr = 0.0
            topfar = 0.0
            topfrr = 0.0

            for nflod in range(1):
                train_list = []
                test_list = []
                for j in range(self.usernum):
                    one = keyer[j]
                    random.shuffle(one)  # 随机打乱list
                    train, test = self.split_list(nflod, one)
                    train_list.append(train)
                    test_list.append(test)
                totalerr = 0.0
                totalfar = 0.0
                totalfrr = 0.0
                for userid in range(self.usernum - 1, self.usernum):
                    print("第%d个用户" % (userid), "固定长度:", self.seq_length, "开始训练。。。")
                    x_train_1 = train_list[userid]
                    nb_word = 0

                    y_train_1 = [1] * len(x_train_1)
                    x_test_1 = test_list[userid]
                    y_test_1 = [1] * len(x_test_1)

                    leave_train = []

                    samplenum = len(y_train_1) * 2

                    print(samplenum)
                    if samplenum % 10 >= 5:
                        samplenum = (int(samplenum / 10) + 1) * 10
                    else:
                        samplenum = int(samplenum / 10) * 10
                    len1 = samplenum - len(y_train_1)

                    for k in range(self.usernum):
                        if k != userid:
                            leave_train = leave_train + train_list[k]
                    print(len1)
                    x_train_1 = x_train_1 + random.sample(leave_train, len1)
                    y_train_1 = y_train_1 + [0] * len1
                    leave_test = []

                    realnum = len(y_test_1)
                    testnum = len(y_test_1) * 2
                    print(realnum)

                    if testnum % 10 >= 5:
                        testnum = (int(testnum / 10) + 1) * 10
                    else:
                        testnum = int(testnum / 10) * 10

                    len2 = testnum - len(y_test_1)

                    for k in range(self.usernum):
                        if k != userid:
                            leave_test = leave_test + test_list[k]
                    print(len2)

                    x_test_1 = x_test_1 + random.sample(leave_test, len2)
                    y_test_1 = y_test_1 + [0] * len2
                    y_test_1_tmp = y_test_1[:]

                    x_train_1 = numpy.array(x_train_1)
                    x_test_1 = numpy.array(x_test_1)
                    y_train_1 = numpy.array(y_train_1)
                    # print(y_train_1)
                    model = Sequential()
                    # model.add(Embedding(nb_word + 1,128,dropout=0.2))

                    model.add(Convolution1D(batch_input_shape=(self.batch_size, x_train_1.shape[1], x_train_1.shape[2]),
                                            filters=self.nb_filter, kernel_size=self.filter_length, padding='valid',
                                            activation='relu'))
                    model.add(Dropout(0.5))
                    model.add(MaxPooling1D(pool_size=self.pool_length))

                    model.add(GRU(32, return_sequences=True, stateful=True))
                    model.add(Dropout(0.5))

                    model.add(GRU(32, return_sequences=False))

                    # model.add(Bidirectional(LSTM(32,return_sequences=True,batch_input_shape=(batch_size, x_train_1.shape[1], x_train_1.shape[2]))))
                    # model.add(Dropout(0.5))

                    model.add(Dense(256, activation='relu'))
                    model.add(Dropout(0.5))
                    # model.add(Flatten())
                    model.add(Dense(1, activation="sigmoid"))
                    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
                    # model.fit(x_train_1, y_train_1,epochs=epochs, batch_size=batch_size,verbose=2,shuffle=False,validation_data=(x_test_1, y_test_1))
                    minerr = 1.0

                    # class_weight={0:0.33,1:0.67}
                    # model.fit(x_train_1, y_train_1, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
                    '''
                    for nums in range(200):
                        model.fit(x_train_1, y_train_1, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
                        model.reset_states()

                    setThr(model, x_test_1, realnum)
                    #model.save('model_2.h5')
                   '''
                    model.fit(x_train_1, y_train_1, batch_size=10, epochs=200,
                              validation_data=(x_test_1, y_test_1), verbose=1, shuffle=True)
                    model.fit(x_test_1, y_test_1, batch_size=10, epochs=200,
                              verbose=1, shuffle=True)
                    m = 'model_' + str(self.usernum) + '.h5'
                    model.save(m)
#k=keystroke_lstm(78)
#k.lstm()