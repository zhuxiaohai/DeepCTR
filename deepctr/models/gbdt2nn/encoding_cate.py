"""
Reference:
https://github.com/motefly/DeepGBM
"""

import pandas as pd
import numpy as np
import category_encoders as ce


class CateEncoder(object):
    def __init__(self, cate_col, nume_col, threshold, thresrate, bins, label):
        self.label_name = label
        # cate_col = list(df.select_dtypes(include=['object']))
        self.cate_col = cate_col
        # nume_col = list(set(list(df)) - set(cate_col))
        self.dtype_dict = {}
        for item in cate_col:
            self.dtype_dict[item] = 'str'
        for item in nume_col:
            self.dtype_dict[item] = 'float'
        self.nume_col = nume_col
        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_col + nume_col)
        self.threshold = threshold
        self.thresrate = thresrate
        self.bins = bins
        # for online update, to do
        self.save_value_filter = {}
        self.save_num_bins = {}
        self.samples = 0

    def save2npy(self, df, out_dir, y=None):
        # if not os.path.isdir(out_dir):
        #     os.mkdir(out_dir)
        result = {'label': [], 'index': [], 'feature_sizes': []}
        result['label'] = y.values if y is not None else df[self.label_name].values
        result['index'] = df[self.cate_col + self.nume_col].values
        for item in self.cate_col + self.nume_col:
            result['feature_sizes'].append(df[item].max() + 1)
        for item in result:
            result[item] = np.array(result[item])
            np.save(out_dir + item + '.npy', result[item])

    def fit_transform(self, X, outPath, y=None, persist=True):
        print('----------------------------------------------------------------------')
        print('Fitting and Transforming.')
        print('----------------------------------------------------------------------')
        df = X.astype(dtype=self.dtype_dict)
        df = df.replace(['nan'], np.nan)
        print('Filtering and fillna features')
        for item in self.cate_col:
            value_counts = df[item].value_counts()
            num = value_counts.shape[0]
            self.save_value_filter[item] = list(
                value_counts[:int(num * self.thresrate)][value_counts > self.threshold].index)
            rm_values = set(value_counts.index) - set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        print('Fillna and Bucketize numeric features')
        for item in self.nume_col:
            q_res = pd.qcut(df[item], self.bins, labels=False, retbins=True, duplicates='drop')
            if q_res[0].isnull().sum() > 0:
                print('train')
                print(item)
            df[item] = q_res[0].fillna(-1).astype('int')
            self.save_num_bins[item] = q_res[1]

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.fit_transform(df)
        if persist:
            if y is not None:
                df[self.label_name] = y.values
            df.to_csv(outPath + 'train_cate.csv', index=False)
            self.save2npy(df, outPath, y)
        else:
            return df

    # for test dataset
    def transform(self, X, outPath, y=None, persist=True):
        print('----------------------------------------------------------------------')
        print('Transforming %s.')
        print('----------------------------------------------------------------------')
        df = X.astype(dtype=self.dtype_dict)
        df = df.replace(['nan'], np.nan)
        print('Filtering and fillna features')
        for item in self.cate_col:
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index) - set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in self.nume_col:
            if pd.cut(df[item], self.save_num_bins[item], labels=False, include_lowest=True).isnull().sum() > 0:
                print('test', item)
            df[item] = pd.cut(df[item], self.save_num_bins[item], labels=False, include_lowest=True).fillna(-1).astype(
                'int')

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)
        if persist:
            if y is not None:
                df[self.label_name] = y.values
            df.to_csv(outPath + 'test_cate.csv', index=False)
            self.save2npy(df, outPath, y)
        else:
            return df

    def predict(self, X):
        df = X.astype(dtype=self.dtype_dict)
        df = df.replace(['nan'], np.nan)
        print('Filtering and fillna features')
        for item in self.cate_col:
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index) - set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in self.nume_col:
            df[item] = pd.cut(df[item], self.save_num_bins[item], labels=False, include_lowest=True).fillna(-1).astype(
                'int')

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)
        return df