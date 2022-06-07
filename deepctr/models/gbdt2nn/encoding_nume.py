"""
Reference:
https://github.com/motefly/DeepGBM
"""

import numpy as np
import category_encoders as ce
import collections
import gc
import pdb


def unpackbits(x, num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    to_and = 2 ** np.arange(num_bits).reshape([1, num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])


class NumEncoder(object):
    def __init__(self, cate_col, nume_col, threshold, thresrate, label):
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
        self.tgt_nume_col = []
        self.encoder = ce.ordinal.OrdinalEncoder(cols=cate_col)
        self.threshold = threshold
        self.thresrate = thresrate
        # for online update, to do
        self.save_cate_avgs = {}
        self.save_value_filter = {}
        self.save_num_embs = {}
        self.Max_len = {}
        self.samples = 0

    def fit_transform(self, X, outPath, y=None, persist=True):
        print('----------------------------------------------------------------------')
        print('Fitting and Transforming.')
        print('----------------------------------------------------------------------')
        df = X.astype(dtype=self.dtype_dict)
        df = df.replace(['nan'], np.nan)
        self.samples = df.shape[0]
        print(df.shape[0])
        print('Filtering and fillna features')
        for item in self.cate_col:
            value_counts = df[item].value_counts()
            num = value_counts.shape[0]
            self.save_value_filter[item] = list(
                value_counts[:int(num * self.thresrate)][value_counts > self.threshold].index)
            rm_values = set(value_counts.index) - set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')
            del value_counts
            gc.collect()

        for item in self.nume_col:
            df[item] = df[item].fillna(df[item].mean())
            self.save_num_embs[item] = {'sum': df[item].sum(), 'cnt': df[item].shape[0]}

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.fit_transform(df)
        for col in self.encoder.cols:
            df[col] = df[col].astype(np.int32)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in self.cate_col:
            feats = df[item].values
            labels = y.values if y is not None else df[self.label_name].values
            feat_encoding = {'mean': [], 'count': []}
            feat_temp_result = collections.defaultdict(lambda: [0, 0])
            self.save_cate_avgs[item] = collections.defaultdict(lambda: [0, 0])
            for idx in range(self.samples):
                cur_feat = feats[idx]
                # smoothing optional
                if cur_feat in self.save_cate_avgs[item]:
                    # feat_temp_result[cur_feat][0] = 0.9*feat_temp_result[cur_feat][0] + 0.1*self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1]
                    # feat_temp_result[cur_feat][1] = 0.9*feat_temp_result[cur_feat][1] + 0.1*self.save_cate_avgs[item][cur_feat][1]/idx
                    feat_encoding['mean'].append(
                        self.save_cate_avgs[item][cur_feat][0] / self.save_cate_avgs[item][cur_feat][1])
                    feat_encoding['count'].append(self.save_cate_avgs[item][cur_feat][1] / idx)
                else:
                    feat_encoding['mean'].append(0)
                    feat_encoding['count'].append(0)
                self.save_cate_avgs[item][cur_feat][0] += labels[idx]
                self.save_cate_avgs[item][cur_feat][1] += 1
            df[item + '_t_mean'] = feat_encoding['mean']
            df[item + '_t_count'] = feat_encoding['count']
            self.tgt_nume_col.append(item + '_t_mean')
            self.tgt_nume_col.append(item + '_t_count')

        print('Start manual binary encode')
        rows = None
        for item in self.nume_col + self.tgt_nume_col:
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1, 1))
            else:
                rows = np.concatenate([rows, feats.reshape((-1, 1))], axis=1)
            del feats
            gc.collect()
        for item in self.cate_col:
            print(item)
            feats = df[item].values
            Max = df[item].max()
            bit_len = len(bin(Max)) - 2
            samples = self.samples
            self.Max_len[item] = bit_len
            res = unpackbits(feats, bit_len).reshape((samples, -1))
            rows = np.concatenate([rows, res], axis=1)
            del feats
            gc.collect()
        trn_y = np.array(y.values).reshape((-1, 1)) if y is not None\
            else np.array(df[self.label_name].values).reshape((-1, 1))
        trn_y = trn_y.astype(np.float32)
        del df
        gc.collect()
        trn_x = np.array(rows).astype(np.float32)
        if persist:
            np.save(outPath + '_features.npy', trn_x)
            np.save(outPath + '_labels.npy', trn_y)
        else:
            return trn_x, trn_y

    # for test dataset
    def transform(self, X, outPath, y=None, persist=True):
        print('----------------------------------------------------------------------')
        print('Transforming.')
        print('----------------------------------------------------------------------')
        df = X.astype(dtype=self.dtype_dict)
        df = df.replace(['nan'], np.nan)
        print(df.groupby('set')['cust_no'].count())
        print('Filtering and fillna features')
        for item in self.cate_col:
            value_counts = df[item].value_counts()
            rm_values = set(value_counts.index) - set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in self.nume_col:
            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']
            df[item] = df[item].fillna(mean)

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)
        for col in self.encoder.cols:
            df[col] = df[col].astype(np.int32)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in self.cate_col:
            avgs = self.save_cate_avgs[item]
            df[item + '_t_mean'] = df[item].map(lambda x: avgs[x][0] / avgs[x][1] if x in avgs else 0)
            df[item + '_t_count'] = df[item].map(lambda x: avgs[x][1] / self.samples if x in avgs else 0)

        samples_temp = df.shape[0]
        print('Start manual binary encode of {} , with len {}'.format(outPath, samples_temp))
        rows = None
        for item in self.nume_col + self.tgt_nume_col:
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1, 1))
            else:
                rows = np.concatenate([rows, feats.reshape((-1, 1))], axis=1)
            del feats
            gc.collect()
        for item in self.cate_col:
            print(item)
            feats = df[item].values
            bit_len = self.Max_len[item]
            res = unpackbits(feats, bit_len).reshape((samples_temp, -1))
            rows = np.concatenate([rows, res], axis=1)
            del feats
            gc.collect()
        vld_y = np.array(y.values).reshape((-1, 1)) if y is not None\
            else np.array(df[self.label_name].values).reshape((-1, 1))
        vld_y = vld_y.astype(np.float32)
        gc.collect()
        vld_x = np.array(rows).astype(np.float32)
        if persist:
            np.save(outPath + '_features.npy', vld_x)
            np.save(outPath + '_labels.npy', vld_y)
        else:
            return vld_x, vld_y

    # for update online dataset
    def refit_transform(self, X, outPath, persist=True):
        print('----------------------------------------------------------------------')
        print('Refitting and Transforming')
        print('----------------------------------------------------------------------')
        df = X.astype(dtype=self.dtype_dict)
        df = df.replace(['nan'], np.nan)
        samples = df.shape[0]

        print('Filtering and fillna features')
        for item in self.cate_col:
            value_counts = df[item].value_counts()

            rm_values = set(value_counts.index) - set(self.save_value_filter[item])
            df[item] = df[item].map(lambda x: '<LESS>' if x in rm_values else x)
            df[item] = df[item].fillna('<UNK>')

        for item in self.nume_col:
            self.save_num_embs[item]['sum'] += df[item].sum()
            self.save_num_embs[item]['cnt'] += df[item].shape[0]
            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']
            df[item] = df[item].fillna(mean)

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)
        for col in self.encoder.cols:
            df[col] = df[col].astype(np.int32)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in self.cate_col:
            feats = df[item].values
            labels = df[self.label_name].values
            feat_encoding = {'mean': [], 'count': []}

            for idx in range(samples):
                cur_feat = feats[idx]

                if self.save_cate_avgs[item][cur_feat][1] == 0:
                    pdb.set_trace()

                feat_encoding['mean'].append(
                    self.save_cate_avgs[item][cur_feat][0] / self.save_cate_avgs[item][cur_feat][1])
                feat_encoding['count'].append(self.save_cate_avgs[item][cur_feat][1] / (self.samples + idx))

                self.save_cate_avgs[item][cur_feat][0] += labels[idx]
                self.save_cate_avgs[item][cur_feat][1] += 1
            df[item + '_t_mean'] = feat_encoding['mean']
            df[item + '_t_count'] = feat_encoding['count']

        self.samples += samples

        print('Start manual binary encode')
        rows = None
        for item in self.nume_col + self.tgt_nume_col:
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1, 1))
            else:
                rows = np.concatenate([rows, feats.reshape((-1, 1))], axis=1)
            del feats
            gc.collect()
        for item in self.cate_col:
            print(item)
            feats = df[item].values

            bit_len = self.Max_len[item]
            res = unpackbits(feats, bit_len).reshape((samples, -1))
            rows = np.concatenate([rows, res], axis=1)
            del feats
            gc.collect()

        vld_y = np.array(df[self.label_name].values).reshape((-1, 1))
        vld_y = vld_y.astype(np.float32)

        del df
        gc.collect()
        vld_x = np.array(rows).astype(np.float32)

        if persist:
            np.save(outPath + '_features.npy', vld_x)
            np.save(outPath + '_labels.npy', vld_y)
        else:
            return vld_x, vld_y

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
            mean = self.save_num_embs[item]['sum'] / self.save_num_embs[item]['cnt']
            df[item] = df[item].fillna(mean)

        print('Ordinal encoding cate features')
        # ordinal_encoding
        df = self.encoder.transform(df)
        for col in self.encoder.cols:
            df[col] = df[col].astype(np.int32)

        print('Target encoding cate features')
        # dynamic_targeting_encoding
        for item in self.cate_col:
            avgs = self.save_cate_avgs[item]
            df[item + '_t_mean'] = df[item].map(lambda x: avgs[x][0] / avgs[x][1] if x in avgs else 0)
            df[item + '_t_count'] = df[item].map(lambda x: avgs[x][1] / self.samples if x in avgs else 0)

        samples_temp = df.shape[0]
        rows = None
        for item in self.nume_col + self.tgt_nume_col:
            feats = df[item].values
            if rows is None:
                rows = feats.reshape((-1, 1))
            else:
                rows = np.concatenate([rows, feats.reshape((-1, 1))], axis=1)
            del feats
            gc.collect()
        for item in self.cate_col:
            feats = df[item].values
            bit_len = self.Max_len[item]
            res = unpackbits(feats, bit_len).reshape((samples_temp, -1))
            rows = np.concatenate([rows, res], axis=1)
            del feats
            gc.collect()
        gc.collect()
        vld_x = np.array(rows).astype(np.float32)
        return vld_x