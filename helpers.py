import gensim
import time
import numpy as np
import collections
import matplotlib.pyplot as plt
import wordcloud
import math



# https://towardsdatascience.com/dont-be-afraid-of-nonparametric-topic-models-part-2-python-e5666db347a
# https://github.com/ecoronado92/towards_data_science/blob/master/hdp_example/scripts/model_funcs.py

def train_model(m, max_iter=1000, perp_threshold=1e-3, window=10):
    """
    Train model 'm' until perplexity varies by less than 'perp_threshold'
    
    - max_iter: maximum number of iterations
    - window: number of last iterations to consider for stopping
    - perp_threshold: stop training when perplexity varies by less than this fraction of the window median
    
    Returns (time spent, iterations)
    """
    
    t0 = time.time()
    q = collections.deque(maxlen=window)
    iterations = 0
    
    for iter in range(window):
        m.train(1)
        iterations += 1
        p = m.perplexity
        q.append(p)
    for iter in range(max_iter - window):
        m.train(1)
        iterations += 1
        p = m.perplexity
        p_med = np.median(np.asarray(q))
        if p > p_med * (1 - perp_threshold): break
        q.append(p)
    return (
        time.time() - t0,
        iterations
    )


def get_topic_words(model, top_n=20):
    '''Wrapper function to extract topics from trained tomotopy HDP model 
    
    ** Inputs **
    model:obj -> trained model
    top_n: int -> top n words in topic based on frequencies
    
    ** Returns **
    topics: dict, topic id -> tuples of top words and associated frequencies 
    '''
    
    topics = {}
    
    for k in range(model.k):
        try:
            # ignore non-assigned topics
            if not model.is_live_topic(k): continue
        except AttributeError: 
            pass        
        topics[k] = model.get_topic_words(k, top_n=top_n)
        
    return topics



def eval_coherence(topics_dict, vocab, documents, coherence_type='c_v'):
    '''Wrapper function that uses gensim Coherence Model to compute topic coherence scores
    
    ** Inputs **
    topic_dict: dict, result of get_topic_words
    vocab: gensim.corpora.Dictionary
    documents: list of list of strings
    coherence_typ: str -> type of coherence value to compute (see gensim for opts)
    
    ** Returns **
    3-tuple,
        float -> coherence value
        #float list -> per-topic coherence
        #float list -> per-topic coherence deviation
    '''
    
    # Build topic list from dictionary
    topic_list=[]
    for k, tups in topics_dict.items():
        topic_tokens = [w for w,p in tups]
        topic_list.append(topic_tokens)
            

    # Build Coherence model
    cm = gensim.models.CoherenceModel(
        topics=topic_list,
        dictionary=vocab,
        texts=documents, 
        coherence=coherence_type,
        processes=16)
    
    return cm.get_coherence()


# FIXME:
# only consider a topic when its score is > some threshold (2x 1/K ?)
def topic_counts(model, top_n):
    """For each topic, count posts where the topic figure in the `top_n` topics
    
    Returns:
    dict of topic_id (int), posts (int)
    """
    counts = {}
    threshold = 2.0 / model.k
    for k in range(model.k):
        counts[k] = 0
    for doc in tqdm(model.docs):
        for k, v in doc.get_topics(top_n=top_n):
            if v < threshold: next
            counts[k] += 1
    return counts

def eval_perplexity(model, documents):
    count = 0
    docs = [model.make_doc(d) for d in documents]
    _, log_likelihoods = model.infer(doc=docs)
    lengths = [len(d) for d in documents]
    
    assert len(lengths) == len(log_likelihoods)
    return math.exp(-np.sum(np.array(log_likelihoods)) / np.sum(np.array(lengths)))



def top_relevant_terms(model, top_n=30, lambda_=0.4):
    """
    Returns a list of `model.k` pairs of (term, relevance).
    Source: relevnce formula from pyLDAvis code, from Sievert & Shirley (2014)
    """
    result = [None] * model.k

    term_proportion = model.vocab_freq / np.sum(model.vocab_freq) # tf -> model.vocab_freq
    for i in range(model.k):
        topic_term_dists = model.get_topic_word_dist(i)
        log_lift = np.log(topic_term_dists / term_proportion)
        log_ttd = topic_term_dists
        relevance = lambda_ * log_ttd + (1 - lambda_) * log_lift
        terms_idx = np.argsort(relevance)[-top_n:]
        result[i] = [(model.vocabs[i], relevance[i]) for i in reversed(terms_idx)]
    
    return result


def make_wordcloud(data, n_words=50, w=600, h=300):

    return wordcloud.WordCloud(
        width = w, height = h, 
        background_color = 'white',
        prefer_horizontal = 1,
        relative_scaling = 1,
        min_font_size = h/40
    ).generate_from_frequencies(data) 

def render_clouds(model, lmbda=0.4, n_words=30):
    fig=plt.figure(figsize=(16, 12 * model.k // 35))
    columns = 5
    rows = math.ceil(model.k / columns)
    
    terms = top_relevant_terms(model, top_n=50)
    words = {} # topic idx -> word -> frequency
    for i in range(model.k):
        words[i] = dict(terms[i])

    images = {} # topic idx -> image
    for i in range(0, model.k):        
        images[i] = make_wordcloud(words[i])
    
    for i in range(model.k):
        img = images[i]
        subp = fig.add_subplot(rows, columns, i+1)
        subp.set_title("Topic #%d" % (i+1,))
#         subp.set_title("Topic #%d (%d%% posts)" % (i, 100 * counts[i] / counts.sum()))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout(pad = 2.0)
    plt.show()
