from sklearn.base import BaseEstimator, TransformerMixin
import re
import pandas as pd
from pyvi import ViTokenizer


class Preprocesser(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def remove_stop_words(self, X):
        with open('/Users/DuyHome/HocTap/DATN/FeatureEngineering/vietnamese-stopwords.txt') as f:
            stop_words = f.readlines()
            stop_words = list(set(m.replace(' ', '_').strip()
                              for m in stop_words))
            for i, doc in enumerate(X):
                arr = []
                try:
                    for word in doc.split():
                        if word not in stop_words and word != " ":
                            arr.append(word)
                except Exception as e:
                    print("stop_word" + str(e) + "and index:" + str(i))
                temp = " ".join(arr).strip()
                X = X.replace(doc, temp)
        return X

    def remove_html(self, X):
        for i, v in enumerate(X):
            try:
                X = X.replace(v, re.sub(r'<[^>]*>', '', v))
            except Exception as e:
                print("html" + str(e) + "and index:" + str(i))
        return X
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    def loaddicchar(self):
        dic = {}
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
            '|')
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
            '|')
        for i in range(len(char1252)):
            dic[char1252[i]] = charutf8[i]
        return dic

    # Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
    def convert_unicode(self, X):
        dic = self.loaddicchar()
        for i, v in enumerate(X):
            try:
                X = X.replace(v, re.sub(
                    r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
                    lambda x: dic[x.group()], v))
            except Exception as e:
                print("convert uni" + str(e) + "and index:" + str(i))
        return X

    def tokenize(self, X):
        for i, v in enumerate(X):
            try:
                X = X.replace(v, ViTokenizer.tokenize(v))
            except Exception as e:
                print("token" + str(e) + "and index:" + str(i))
        return X

    def lower_key(self, X):
        for i, v in enumerate(X):
            try:
                X = X.replace(v, v.lower())
            except Exception as e:
                print("lower key" + str(e) + "and index:" + str(i))
        return X

    def remove_noise_key(self, X):
        for i, v in enumerate(X):
            try:
                # xóa các ký tự không cần thiết
                z = re.sub(
                    r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', v)
                # g = re.sub(r'[.]', ' ', z)
                X = X.replace(v, re.sub(r'\s+', ' ', z).strip())
            except Exception as e:
                print("remove noise" + str(e) + "and index:" + str(i))
        return X

    def remove_digits(self, X):
        for i, v in enumerate(X):
            try:
                X = X.replace(v, re.sub(r'\b\d+\b', "", v))
            except Exception as e:
                print("remove digit" + str(e) + "and index:" + str(i))
        return X

    def remove_whitespace(self, X):
        for i, v in enumerate(X):
            try:
                arr = []
                for i in v.split():
                    if i != " ":
                        arr.append(i)
                content = " ".join(arr)
                X = X.replace(v, content)
            except Exception as e:
                print("remove whitespace" + str(e) + "and index:" + str(i))
        return X

    def remove_punctuation(self, X):
        for i, v in enumerate(X):
            try:
                X = X.replace(
                    v, re.sub(r'[!"#$%&()*+,\-./:;<=>?@[\\\]^`{|}~\t\n1234567890]', '', v))
            except Exception as e:
                print("remove punctuation" + str(e))
        return X

    def remove_word_with_length_of_one(self, X):
        for i, v in enumerate(X):
            try:
                new_docs = [word for word in v.split() if len(word) > 1]
                X = X.replace(
                    v, " ".join(new_docs).strip())
            except Exception as e:
                print("remove one length" + str(e))
        return X

    def transform(self, X):
        X = pd.Series(X)
  
        X = self.remove_html(X)
        X = self.convert_unicode(X)
        X = self.tokenize(X)
        X = self.lower_key(X)
        X = self.remove_noise_key(X)
        X = self.remove_digits(X)
        X =  self.remove_word_with_length_of_one(X)
        # remove stop word
        # X = self.remove_stop_words(X)
        # remote whitespace
        # X = self.remove_whitespace(X)
        return X
