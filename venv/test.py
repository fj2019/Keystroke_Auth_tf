# coding:utf-8
import numpy
import PyHook3
import pythoncom
from time import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import LSTM, GRU
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import load_model
from model_test import model_test
from keystroke_lstm_0 import keystroke_lstm
import win32api
from flask import flash
from keras import backend as K
import time
#K.clear_session()

class keycol:

    keylist_base=[]
    d=[]
    is_imp=0
    model = load_model('model_1.h5')
    model.predict(numpy.array([[[0]*6]*30]*10))
    upnum=0
    def onKeyboardEvent(self,event):
        if event.MessageName == 'key up':
            self.upnum+=1
        #print(111)
        print (event.Key, event.KeyID, event.MessageName, event.Time)
        tmp=[]
        #tmp.append(event.Key)
        tmp.append(int(event.KeyID))
        key=event.MessageName.split(' ')
        if len(key) == 3:
            tmp.append(key[0].capitalize() + key[2].capitalize())
        else:
            tmp.append(key[0].capitalize() + key[1].capitalize())
        #tmp.append(event.MessageName)
        tmp.append(float(event.Time))
        self.keylist_base.append(tmp)
        lengt=len(self.keylist_base)
        if self.upnum%51==0 and self.upnum!=0 and event.MessageName == 'key up':
            keytimelist = [[] for i in range(int(lengt / 2) + 10)]
            d=[]
            num=0
            for i in range(lengt):
                keylist=self.keylist_base[i]
                #print(keylist)
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
                        if keylist[1] == "KeyUp":
                            
                            keytimelist[d[f][3]].append(keylist[0])
                            keytimelist[d[f][3]].append(d[f][2])
                            keytimelist[d[f][3]].append(keylist[2])
                            del d[f]

            #print(keytimelist)
            for i in range(len(keytimelist) - 1, -1, -1):
                if len(keytimelist[i]) == 0:
                    keytimelist.pop(i)
            #print(keytimelist)
            key_word_list=[]
            keylist1=keytimelist
            for i in range(len(keylist1) - 1):
                key_word = []
                key_word.append(keylist1[i][0])
                key_word.append(keylist1[i + 1][0])
                # key_word.append(keylist[i+2][0])

                key_word.append(keylist1[i][2] - keylist1[i][1])
                key_word.append(keylist1[i + 1][2] - keylist1[i + 1][1])
                # key_word.append(keylist[i+2][2]-keylist[i+2][1])

                key_word.append(keylist1[i + 1][1] - keylist1[i][2])
                key_word.append(keylist1[i + 1][1] - keylist1[i][1])
                key_word_list.append(key_word)
            #print(key_word_list)
            #print(len(key_word_list))
            seq_length=30
            l = 0
            r = seq_length
            # print(r)
            len1 = len(key_word_list)
            xx = []
            while r < len1:
                xx.append(key_word_list[l:r])
                l = l + 1
                r = r + 1
            if r > len1:
                xx.append(key_word_list[len1 - seq_length:len1])
            testlist = xx
            #print(len(testlist))
            len1 = len(testlist)
            testlist = numpy.array(testlist)
            #with self.graph.as_default():
            #print(time.time())
            res=self.model.predict(testlist, batch_size=10)
            #print(time.time())
            print(res)
            print(len(res))
            num = 0
            for i in range(len1):
                if res[i] < 0.5:
                    num = num + 1
            print(num / len1)
            if num >= int(0.5 * len1):
                print("经系统检测，存在入侵可能请输入验证码！！")
                win32api.PostQuitMessage()
                #强制中断用户操作,转入验证
            else:
                print("账户安全！！")
            self.keylist_base=[]
            # m=model_test(testlist,len1)
            # if m.is_intruder==1:
            #      flash('经系统检测，存在入侵可能请输入验证码！！')
            # else:
            #      flash('账户安全！！')

        # 监听键盘事件
        return True

    def main(self):
        hm = PyHook3.HookManager()
        # 监听所有的键盘事件
        # hm.KeyDown = onKeyboardEvent
        hm.KeyDown = self.onKeyboardEvent
        hm.KeyUp = self.onKeyboardEvent
        # 设置键盘”钩子“aaaaaaaaaaaaaaaaaaaa
        hm.HookKeyboard()
        # 进入循环侦听，需要手动进行关闭，否则程序将一直处于监听的状态。可以直接设置而空而使用默认值
        pythoncom.PumpMessages()
        # 我也不知道为什么直接放置到main函数中不管

class keycol_train:
    keylist_base=[]
    userid=76
    txtid=0
    #upnum=0
    def onKeyboardEvent(self,event):
        #print(111)
        tmp=[]
        tmp.append(event.Key)
        tmp.append(event.KeyID)
        tmp.append(event.MessageName)
        tmp.append(event.Time)
        print (event.Key, event.KeyID, event.MessageName, event.Time)
        self.keylist_base.append(tmp)
        if event.Key == 'Oem_2' and event.MessageName == 'key up':
            self.txtid = self.txtid + 1
            file='0'+str(self.userid)+'.'+str(self.txtid)
            f = open('free_txt/baseline_0/'+file+'.txt', 'a')
            for i in range(len(self.keylist_base)):
                tmp=self.keylist_base[i]
                f.write(tmp[0]+' ')
                f.write(str(tmp[1])+' ')
                f.write(tmp[2]+' ')
                f.write(str(tmp[3])+'\n')
            f.close()
            if self.txtid==3:
                win32api.PostQuitMessage()
                #print(1)
                self.txtid = 0
                l=keystroke_lstm()
                l.lstm()

            self.keylist_base=[]
        return True

    def main(self):
        hm = PyHook3.HookManager()
        # 监听所有的键盘事件
        # hm.KeyDown = onKeyboardEvent
        hm.KeyDown = self.onKeyboardEvent
        hm.KeyUp = self.onKeyboardEvent
        # 设置键盘”钩子“aaaaaaaaaaaaaaaaaaaa
        hm.HookKeyboard()
        # 进入循环侦听，需要手动进行关闭，否则程序将一直处于监听的状态。可以直接设置而空而使用默认值
        pythoncom.PumpMessages()
        # 我也不知道为什么直接放置到main函数中不管