#This is one more implementation for the same EMNLP problem set Named entity recognition for fatal shooting.
#Here I have used a slightly different approach for the same task.
#RNN and lstm code is taken from https://github.com/lipiji/rnn-theano 
cudaid = 1 
import os
os.environ["THEANO_FLAGS"] = "device=cuda" + str(cudaid) 

import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *
import data


def dotlog_reg(r, s):
    # similar to np.dot(r, np.log(s)) but will deal with cases where s=0
    assert len(r) == len(s)
    ent = 0
    for i in range(len(r)):
        if s[i] == 0.0: continue
        if np.isnan(s[i]): continue
        if np.isnan(r[i]): continue
        ent += r[i]*np.log(s[i])
    return ent

def elbo(q, preds):
    '''
    q: this is the marginal posterior of each mention label q(z_i=1)=p(z_i =1 | x, y)
    preds: this is the prediction given by the training function (LR or CNN)
        P(z_i | x)
    '''
    assert q.shape[0] == preds.shape[0]
    #all predictions should be non-zero, otherwise we're calculating the cross-entropy incorrectly
    assert np.all(preds != 0.0)
    elbo = 0
    #----z_i =1, we don't need to change q
    #weighted log likelihood (or neg. cross entropy) when z_i=1
    #elbo += np.dot(q, np.log(preds))
    elbo += dotlog_reg(q, preds)

    #(neg) entropy when z_i = 1
    #elbo -= np.dot(q, np.log(q))
    elbo -= dotlog_reg(q, q)
    #----z_i = 0 we need to take 1 - q
    #weighted log likelihood (or neg. cross entropy) when z_i=0
    #elbo += np.dot(1- q, np.log(1 - preds))
    elbo += dotlog_reg(1-q, 1-preds)
    #(neg) entropy when z_i =0
    #elbo -= np.dot(1 - q, np.log(1- q))
    elbo -= dotlog_reg(1-q, 1-q)
    entropy = dotlog_reg(q, q) + dotlog_reg(1-q, 1-q)
    e_neg_ll = dotlog_reg(q, preds) + dotlog_reg(1-q, 1-preds)
    return elbo, entropy, e_neg_ll

def read_file(file_name):
    #reads in .json train data
    Y = [] #the pseudolabel classes
    info = [] #info that will be printed for this evaluation
    sents = []
    E = [] #entity ID for each mention
    Epos = {} #whether each entity is positive or not.
    #with codecs.open(file_name, 'r', 'utf-8') as pf:
    with open(file_name, "r") as r:
        for line in r:
            docid = json.loads(line)["docid"].decode('utf-8')
            plabel = float(json.loads(line)["plabel"])
            assert type(plabel) == float
            assert plabel == 0.0 or plabel == 1.0
            name = json.loads(line)["name"].decode('utf-8')
            sent = json.loads(line)["sent_alter"].decode('utf-8')
            info.append({"id": docid, "name": name})
            #info.append({"id": docid, "name": name, "sent": sent, "plabel": plabel})
            Y.append(plabel)
            E.append(name)
            if plabel == 1.0: Epos[name] = True
            else: Epos[name] = False
            sents.append(sent)
    Y = np.array(Y)
    assert len(Y) == len(info) == len(sents) == len(E)
    print("READ FILE")
    print(("NUM SENTS in {0}: {1}".format(file_name, len(sents))))
    return sents, Y, info, E, Epos

def init_curr_preds(Nmention, E, Epos):
    eid2rows = defaultdict(list) #keys = entities, values = list of sentence numbers, ex: {"John Doe": [0, 3, 6 ...]}
    for i in range(Nmention):
        eid2rows[E[i]].append(i)
    eid2rows = {eid: np.array(inds, dtype=int) for (eid,inds) in list(eid2rows.items())}
    # assert min(eid2rows)==0 #KAK: don't think this assertion is what we want, keys of eid2rows are names
    all_values = np.concatenate(list(eid2rows.values()))
    assert min(all_values)==0
    # assert max(eid2rows)==len(eid2rows)-1 #KAK: again, don't think this assertion is what we want, keys of eid2rows are names
    assert max(all_values)==Nmention-1
    assert len(Epos)==len(eid2rows)

    print(("%s mentions, %s entities" % (Nmention, len(eid2rows))))

    # Initialize to pseudolabel
    init_preds = np.zeros(Nmention)
    for eid in eid2rows:
        if Epos[eid]:
            init_preds[eid2rows[eid]] = 1.0
    init_preds[init_preds > 1-1e-16] = 1-1e-16
    init_preds[init_preds < 1e-16] = 1e-16
    return init_preds, eid2rows


exec(compile(open("conv_net_classes.py").read(), "conv_net_classes.py", 'exec'))

import argparse

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)


def disj_inference(preds, E, Epos, eid2rows):
    # E: length Nmention. entity ID for each mention
    # Epos: length Nentity.  whether each entity is positive or not.
    # preds: the P(z_i=1|x_i) prior probs for each mention
    Nmention = len(preds); assert Nmention==len(E)
    assert len(eid2rows)==len(Epos)
    ## not sure if this will be necessary. could be.
    #preds[preds > 1-1e-5] = 1-1e-5
    #preds[preds < 1e-5] = 1e-5
    assert np.all(preds >= 0) and np.all(preds <= 1)

    ret_marginals = np.zeros(Nmention)
    for (eid,row_indices) in list(eid2rows.items()):
        if not Epos[eid]:
            # set Q=0 for these cases. use initialization from above
            continue
        ent_ment_preds = preds[row_indices]
        marginals = infer_disj_marginals(ent_ment_preds)
        ret_marginals[row_indices] = marginals

    # more numeric insanity
    ret_marginals[np.where(ret_marginals > 1.0)] = 1.0
    ret_marginals[np.where(ret_marginals < 0.0)] = 0.0

    return np.nan_to_num(ret_marginals) # more NAN fixes

def infer_disj_marginals(priors):
    return infer_disj_marginals2(priors)

def infer_disj_marginals2(priors):
    disj_prob = 1-np.prod(1-priors)
    if disj_prob==0: return priors
    return priors / disj_prob

def pad_weights(ts, w):
    '''ts: training set, w: weights'''
    padding = len(ts[:,-1]) - len(w)
    assert len(ts[:,-1]) == len(w) + padding
    weights2 = np.lib.pad(w, (0,padding), 'constant', constant_values=(0, 0))
    return weights2



def train_conv_net(fold_i,
                   datasets,
                   trackers,
                   U,
                   img_w=300,
                   filter_hs=[3,4,5],
                   hidden_units=[100,2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,
                   pickle_w=False,
                   mode="vanilla",
                   cur_preds=None,
                   E=None,
                   Epos = None,
                   eid2rows=None):

    e = 0.01
    lr = 2e-5
    drop_rate = 0.5
    batch_size = 32

    x = T.matrix('x')
    y = T.ivector('y')
    wt = T.ivector('wt')
    idxs = T.ivector()
    Words = theano.shared(value = U, name = "Words")


    hidden_size = [100, 100]
    # try: gru, lstm
    cell = "lstm"
    # try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
    optimizer = "adadelta" 

    #img_h = len(datasets[0][0])-1
    dim_x = len(datasets[0][0])-1
    dim_y =2
    # seqs, i2w, w2i, data_xy = data.char_sequence("/data/toy.txt", batch_size)
    # dim_x = len(w2i)
    # dim_y = len(w2i)
    #print "#features = ", dim_x, "#labels = ", dim_y

    print "compiling..."
    model = RNN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate)

    train = [o for o in open("train.json")]
        #uncomment this back on for EM
    Q = disj_inference(cur_preds, E, Epos, eid2rows)
    weights = np.concatenate((Q, 1-Q))
        #define parameters of the model and update functions using adadelta
    params = model.params

    cost = model.train(X, maskX, Y, maskY, lr, local_batch_size)
    #dropout_cost = model.dropout_negative_log_likelihood(y,wt)
    grad_updates = model.gparams
    train_set = datasets[0]

    for i in range(len(train)):
        train_set[i][len(train_set[i]) - 1] =1
        train_set[i + len(train)][len(train_set[i]) - 1] =0

        # check that weights match vanilla
    pos = [o for o in np.where(weights > .01)[0] if o < len(train)]
    neg = [o - len(train) for o in np.where(weights > .01)[0] if o > len(train)] #for u in np.where(weights > .01):
    for p in pos:
            #print p
        assert json.loads(train[p])["plabel"] == 1
    for n in neg:
        assert json.loads(train[n])["plabel"] == 0
    weights[np.where(weights > .99)] = 1
    weights[np.where(weights < .01)] = 0
    train_set_orig = np.where(weights > .99)
        #weights = np.ones(len(train_set))
    for lno,ln in enumerate(train_set):
        if lno < len(train_set)/2: 
            endx = len(ln) -1
            mirror = train_set[lno + len(train)]
            assert np.array_equal(ln[0:len(ln)-1], mirror[0:len(ln)-1])
    extra_data_num = batch_size - train_set[0].shape[0] % batch_size
    extra_data = train_set[:extra_data_num]
    new_data=np.append(train_set,extra_data,axis=0)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches))

    test_set_x = datasets[1][0][:,:img_h]  #TODO THIS 0 index only gets first of these
    test_set_y = np.asarray(datasets[1][0][:,-1],"int32") #TODO THIS 0 index only gets first of these

    train_batches = {}
    for i in range(len(datasets[2])):
        train_batches[i] = datasets[2][i][:,:dim_x]  #fill a bunch of batches

    train_set = new_data[:n_train_batches*batch_size,:]

    weights = pad_weights(train_set, weights)
    permuted_weights = weights


    train_set_x, train_set_y, train_set_wt = shared_dataset((train_set[:,:img_h],train_set[:,-1], permuted_weights))
    n_val_batches = n_batches - n_train_batches

    #compile theano functions to get train/val/test errors
    test_model = theano.function([idxs], classifier.errors(y),
             givens={
                 x: train_set_x[idxs],
                 y: train_set_y[idxs]},
                                 allow_input_downcast=True)
    train_model = theano.function([idxs, wt], cost, updates=grad_updates,
          givens={
              x: train_set_x[idxs],
              y: train_set_y[idxs]},
                                  allow_input_downcast = True)
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))#Cast any tensor x to a Tensor of the same shape, but with a different numerical type dtype.




print "training..."
start = time.time()
g_error = 9999.9999
for i in xrange(200):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in data_xy.items():
        X = xy[0] 
        Y = xy[1]
        maskX = xy[2]
        maskY = xy[3]
        local_batch_size = xy[4]
        cost = model.train(X, maskX, Y, maskY, lr, local_batch_size)
        error += cost
    in_time = time.time() - in_start

    error /= len(data_xy);
    if error < g_error:
        g_error = error
        #save_model("./model/rnn.model_" + str(i), model)

    print "Iter = " + str(i) + ", Loss = " + str(error) + ", Time = " + str(in_time)
    if error <= e:
        break

print "Finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/char_rnn.model", model)



def run():
    print("loading data...")
    with gzip.open('mr.p', 'rb') as f:
        x = pickle.load(f)
    with gzip.open("datasets.p", "rb") as f:
        datasets = pickle.load(f)
    with gzip.open("trackers.p", "rb") as f:
        trackers = pickle.load(f)
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print("data loaded!")
    mode = sys.argv[1]
    assert mode == "em" or mode == "vanilla" or mode == "debug"
    non_static=True
    exec(compile(open("conv_net_classes.py").read(), "conv_net_classes.py", 'exec'))
    U = W
    pickle_w = False
    results = []
    r = list(range(0,1))
    max_l = [kk.replace("\n", "") for kk in open("max_l.txt")].pop()
    max_l = int(max_l)
    i = 0
    d = .5

    sents_train, Y_train, info_train, E, Epos  = read_file("train.json")
    if mode == "EM":
        em_iter = [l for l in open('em.iter')].pop()
        em_iter = int(em_iter)
        init_preds, eid2rows = init_curr_preds(len(sents_train),E, Epos)
    else:
        em_iter = "NA"
        sents_train, Y_train, info_train, E_train, Epos_train  = read_file("train.json")
        init_preds, eid2rows = init_curr_preds(len(sents_train), E, Epos)
    s = 9
    perf = train_conv_net(i, datasets,trackers,
                          U,
                          lr_decay=0.95,
                          filter_hs=[3,4,5],
                          conv_non_linear="relu",
                          hidden_units=[100,2],
                          shuffle_batch=True,
                          n_epochs=100,
                          sqr_norm_lim=s,
                          non_static=non_static,
                          pickle_w=pickle_w,
                          batch_size=50,
                          dropout_rate=[d],
                          mode=mode,
                          cur_preds=init_preds,
                          E=E,
                          Epos=Epos,
                          eid2rows=eid2rows
                          )
    print(("cv: " + str(i) + ", perf: " + str(perf), ", dropout_rate:", d))
    results.append(perf)

if __name__=="__main__":
    run()

