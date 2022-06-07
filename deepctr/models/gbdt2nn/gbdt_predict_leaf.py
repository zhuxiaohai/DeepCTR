import numpy as np
from .tree_interpreter import ModelInterpreter


def SubGBDTLeaf_cls(x, gbm, maxleaf, num_slices, args):
    # total feature_num
    MAX=x.shape[1]
    # (None, number of trees)
    leaf_preds = gbm.predict(x, pred_leaf=True).reshape(x.shape[0], -1)
    n_trees = leaf_preds.shape[1]
    # number of trees in a group
    step = int((n_trees + num_slices - 1) // num_slices)
    step = max(step, 1)
    # get the output of all the trees, and save it to array (number of trees, maxleaf)
    leaf_output = np.zeros([n_trees, maxleaf], dtype=np.float32)
    for tid in range(n_trees):
        num_leaf = np.max(leaf_preds[:,tid]) + 1
        for lid in range(num_leaf):
            leaf_output[tid][lid] = gbm.get_leaf_output(tid, lid)
    rest_nt = n_trees
    # specify the number of trees and n_feature used in a group (n_feature is the number of features specified for a group)
    modelI = ModelInterpreter(gbm, args)
    if args['group_method'] == 'Equal' or args['group_method'] == 'Random':
        clusterIdx = modelI.EqualGroup(num_slices, args)
        n_feature = args['feat_per_group']
    treeI = modelI.trees
    # rand = (args.group_method == 'Random')
    # iter over every group
    for n_idx in range(num_slices):
        # get trees in a group
        tree_indices = np.where(clusterIdx == n_idx)[0]
        trees = {}
        tid = 0
        for jdx in tree_indices:
            trees[str(tid)] = treeI[jdx].raw
            tid += 1
        tree_num = len(tree_indices)
        layer_num = 1
        xi = []
        xi_fea = set()
        # get feature importance(gain) of the features used in all the trees in a group
        all_hav = {} # set([i for i in range(MAX)])
        for jdx, tree in enumerate(tree_indices):
            for kdx, f in enumerate(treeI[tree].feature):
                if f == -2:
                    continue
                if f not in all_hav:
                    all_hav[f] = 0
                all_hav[f] += treeI[tree].gain[kdx]
        used_features = []
        rest_feature = []
        # sort according to feature importance
        all_hav = sorted(all_hav.items(), key=lambda kv: -kv[1])
        # get the first n_feature (n_feature is the number of features specified for a group) to be used in a group
        used_features = [item[0] for item in all_hav[:n_feature]]
        # if rand:
        # used_features = np.random.choice(MAX, len(used_features), replace = False).tolist()
        used_features_set = set(used_features)
        # if the features used in a group is less than n_features, fill with psudo-features, which are all-zeros
        for kdx in range(max(0, n_feature - len(used_features))):
            used_features.append(MAX)
        # (None, number of trees in a group) with values bing leaf index
        cur_leaf_preds = leaf_preds[:, tree_indices]
        # get the output of a group of trees for every sample: (None,)
        new_y = np.zeros(x.shape[0])
        # to be specific: add up the output of every tree in a group
        for jdx in tree_indices:
            # get the output of tree "jdx" for every sample: (None,)
            new_y += np.take(leaf_output[jdx,:].reshape(-1), leaf_preds[:,jdx].reshape(-1))
        new_y = new_y.reshape(-1, 1).astype(np.float32)
        # yield a group
        yield used_features, new_y, cur_leaf_preds, np.mean(np.take(leaf_output, tree_indices,0)), np.mean(leaf_output)


def gbdt_predict(x, gbm, args):
    gbms = SubGBDTLeaf_cls(x, gbm, args['maxleaf'], num_slices=args['nslices'], args=args)
    min_len_features = x.shape[1]
    used_features = []
    tree_outputs = []
    leaf_preds = []
    max_ntree_per_split = 0
    group_average = []
    for used_feature, new_y, leaf_pred, avg, all_avg in gbms:
        used_features.append(used_feature)
        min_len_features = min(min_len_features, len(used_feature))
        tree_outputs.append(new_y)
        leaf_preds.append(leaf_pred)
        group_average.append(avg)
        max_ntree_per_split = max(max_ntree_per_split, leaf_pred.shape[1])
    for i in range(len(used_features)):
        used_features[i] = sorted(used_features[i][:min_len_features])
    n_models = len(used_features)
    group_average = np.asarray(group_average).reshape(n_models, 1, 1)
    for i in range(n_models):
        if leaf_preds[i].shape[1] < max_ntree_per_split:
            # leaf index is offset by 1, so 0 means padding
            leaf_preds[i] = np.concatenate([leaf_preds[i] + 1,
                                            np.zeros([leaf_preds[i].shape[0],
                                                      max_ntree_per_split-leaf_preds[i].shape[1]],
                                                     dtype=np.int32)], axis=1)
    leaf_preds = np.concatenate(leaf_preds, axis=1)
    return leaf_preds, tree_outputs, group_average, used_features, n_models, max_ntree_per_split, min_len_features


