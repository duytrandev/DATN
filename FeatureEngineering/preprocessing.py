from sklearn.base import BaseEstimator, TransformerMixin
from underthesea import word_tokenize
import re
import pandas as pd


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
                    for word in doc.split(' '):
                        if word not in stop_words:
                            arr.append(word)
                except Exception as e:
                    print("stop_word" + str(e) + "and index:" + str(i))
                temp = " ".join(arr)
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
                X = X.replace(v, word_tokenize(v, format="text"))
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
                X = X.replace(v, re.sub(
                    r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]', ' ', v))
                # xóa khoảng trắng thừa
                X = X.replace(v, re.sub(r'\s+', ' ', v).strip())
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

    def removeNaAndDuplicates(self, X, subset):
        X = pd.Series(X)
        X = X.dropna()
        X = X.drop_duplicates(subset=subset)
        return X

    def removeOutliersIQR(self, X):
        q1 = X["Length"].quantile(0.25)

        q3 = X["Length"].quantile(0.75)
        iqr = q3 - q1

        # Identify outliers as those that fall outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        outliers = X[(X < q1 - 1.5*iqr) | (X > q3 + 1.5*iqr)]

        # Remove outliers from the original DataFrame
        return X[(X >= q1 - 1.5*iqr) & (X <= q3 + 1.5*iqr)]

    def transform(self, X):
        X = pd.Series(X)
        X = self.remove_digits(X)
        # xóa html code
        X = self.remove_html(X)
        # chuẩn hóa unicode
        X = self.convert_unicode(X)
        # tách từ
        X = self.tokenize(X)
        # đưa về lower
        X = self.lower_key(X)
        # remove unneecessary key
        X = self.remove_noise_key(X)
        # remove stop word
        X = self.remove_stop_words(X)
        return X