import os
import numpy as np
import pandas as pd
from collections import deque
import tensorflow as tf
import threading
import queue


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def calculate_hit(sorted_list,topk,true_items,rewards,r_click,total_reward,hit_click,ndcg_click,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)
                   
# no use, mix up
def mix_up(self, state, target, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, 0.1)
    else:
        lam = 1
    tmp = tf.concat([state, tf.expand_dims(tf.cast(target, dtype=tf.float32), axis=1)], axis=1)
    tmp  = tf.gather(tmp, tf.random.shuffle(tf.range(tf.shape(tmp)[0])))
    state_ = tmp[:, :-1]
    target_b = tf.squeeze(tf.cast(tmp[:, -1], dtype=tf.int32))        
    state_mixed = lam * state + (1-lam) * state_
    target_a = target

    return state_mixed, target_a, target_b, lam


# class Memory():
#     def __init__(self):
#         self.buffer = deque()
#
#     def add(self, experience):
#         self.buffer.append(experience)
#
#     def sample(self, batch_size):
#         idx = np.random.choice(np.arange(len(self.buffer)),
#                                size=batch_size,
#                                replace=False)
#         return [self.buffer[ii] for ii in idx]

# For REM
def make_coeff(num_heads):
    arr = np.random.uniform(low=0.0, high=1.0, size=num_heads)
    arr /= np.sum(arr)
    return arr.astype(np.float32)

def entropy_correct_replay(replay_buffer):
    
    def _sample(replay_buffer):
        batch = replay_buffer.sample(n=512).to_dict()
        labels = []
        for alist in batch['state'].values():
            for item in alist:
                labels.append(item)
        labels = list(set(labels))
        tmp = pd.DataFrame(data = batch['state'].values())
        entropy = 0.0
        for label in labels:
            # print(np.array(batch['state'] == label).sum())
            prob = np.array(tmp == label).sum() / (512 * 10)
            entropy -= prob * np.log2(prob + 1e-232)
        return entropy, batch

    entropy, batch = _sample(replay_buffer)
    while(np.abs(entropy - 5.513) > 0.202):
        entropy, batch = _sample(replay_buffer)
    return batch
