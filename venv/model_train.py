from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from keras.layers import LSTM, GRU
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import load_model
import numpy
class model_test:
    def __init__(self,X):
        print(1111)
        model=load_model('model_77.h5')
        res=model.predict(X,batch_size=10)
        print(res)
        num=0
        for i in range(len):
            if res[i]<0.5:
                num=num+1
        print(num/len)
        if num>=int(0.15*len):
            self.is_intruder=1
        if self.is_intruder==1:
            print("经系统检测，存在入侵可能请输入验证码！！")
        else:
            print("账户安全！！")
