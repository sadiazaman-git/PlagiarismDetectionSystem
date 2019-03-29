import textract
import os
import nltk
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer, word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams
from django.conf import settings
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag.stanford import StanfordNERTagger
from nltk.corpus import wordnet
from nltk.corpus import verbnet as vn
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pylab
import PIL
import PIL.Image
import io
from io import *
from django.http.response import HttpResponse
from nltk.internals import _java_options

# parser, classifier, jdk
os.environ['STANFORD_PARSER'] = 'C:/python/Lib/site-packages/stanford-parser/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'C:/python/Lib/site-packages/stanford-parser/stanford-english-corenlp-models.jar'
os.environ['JAVA_HOME']= 'C:/Program Files/Java/jdk-10.0.1/bin/java.exe'
os.environ['STANFORD_CLASSIFIER'] = 'C:/python/Lib/site-packages/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'
os.environ['STANFORD_NER_PATH'] = 'C:/python/Lib/site-packages/stanford-ner/stanford-ner.jar'
nltk.internals.config_java(options='-xmx2G')


def plag(p1, p2):
    data1 = textract.process(settings.MEDIA_ROOT+'/'+p1, encoding='utf_8')
    data2 = textract.process(settings.MEDIA_ROOT+'/'+p2, encoding='utf_8')
    readme = data1.decode("utf_8")
    readmee = data2.decode("utf_8")
    for words1 in '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~':
        readme = readme.replace(words1, '')
    readme = readme.replace('\n', ' ')
    for words2 in '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~':
        readmee = readmee.replace(words2, '')
    readmee = readmee.replace('\n', ' ')
    lem = WordNetLemmatizer()
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    words1 = word_tokenize(readme)                              #tokenization
    words2 = word_tokenize(readmee)
    words1 = [lem.lemmatize(word) for word in words1]           #lemmatization
    words2 = [lem.lemmatize(word) for word in words2]
    # words1 = [ps.stem(word)for word in words1]
    # words2 = [ps.stem(word)for word in words2]
    words1 = [word.lower() for word in words1]                  #lowercasing
    words2 = [word.lower() for word in words2]
    source = [w for w in words1 if w not in stop_words]         #stopwords removal
    suspecious = [w for w in words2 if w not in stop_words]
    source = pos_tag(source)                                    #POS tagging
    suspecious = pos_tag(suspecious)
    chunkGram = r"""Chunk: {<NP.?>*<NN.?>*<NNS.?>*<NNP.?>*<PP.?>*<VB.?>*<VP.?>*<SBAR.?>}"""
    ChunkParser = nltk.RegexpParser(chunkGram)                  #chunking
    chunk1 = ChunkParser.parse(source)
    chunk2 = ChunkParser.parse(suspecious)
    file1 = str(chunk1)
    file2 = str(chunk2)
    tree = nltk.Tree.fromstring(file1, read_leaf=lambda x: x.split("/")[0])
    tree = tree.leaves()
    trigrams = list(ngrams(tree, 3))
    tree1 = nltk.Tree.fromstring(file2, read_leaf=lambda x: x.split("/")[0])
    tree1 = tree1.leaves()
    trigrams1 = list(ngrams(tree1, 3))

    def flatten(alist):
        newlist = []
        for item in alist:
            if isinstance(item, list):
                newlist = newlist + flatten(item)
            else:
                newlist.append(item)
        return newlist

    # dependancy relation extraction of file 1

    def dependency_relation1(f1):
        dep_parser = StanfordDependencyParser(os.environ['STANFORD_PARSER'],
                                              os.environ['STANFORD_MODELS'],
                                              model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        sentences = []
        for read in sent_tokenize(f1):
            read = pos_tag(read.split())
            sentences.append(read)
        parse = sum([[list(parse.triples()) for parse in dep_graphs] for dep_graphs in
                     dep_parser.tagged_parse_sents((sentences))], [])
        parse = [item for sublist in parse for item in sublist]
        return parse

    # dependancy relation extraction of file 2

    def dependency_relation2(f2):
        dep_parser = StanfordDependencyParser(os.environ['STANFORD_PARSER'], os.environ['STANFORD_MODELS'],
                                              model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
        sentences = []
        for read in sent_tokenize(f2):
            read = pos_tag(read.split())
            sentences.append(read)
        parse = sum([[list(parse.triples()) for parse in dep_graphs] for dep_graphs in
                     dep_parser.tagged_parse_sents((sentences))], [])
        parse = [item for sublist in parse for item in sublist]
        return parse

    parse = dependency_relation1(readme)
    parse1 = dependency_relation2(readmee)

#name Entity recognizer
    def Name_Entity_recognition1(f1):
        st = StanfordNERTagger(os.environ['STANFORD_CLASSIFIER'],
                               os.environ['STANFORD_NER_PATH'],
                               encoding='utf-8')
        st.java_options ='-mx1000m'
        word = word_tokenize(f1)
        classified_text = st.tag(word)
        return classified_text
#name Entity recognizer

    def Name_Entity_recognition2(f2):
        st = StanfordNERTagger(os.environ['STANFORD_CLASSIFIER'],
                               os.environ['STANFORD_NER_PATH'],
                               encoding='utf-8')
        st.java_options ='-mx1000m'
        word = word_tokenize(f2)
        classified_text = st.tag(word)
        return classified_text

    classify = Name_Entity_recognition1(readme)
    classify1 = Name_Entity_recognition2(readmee)

    def get_continuous_chunks(tagged_sent):
        continuous_chunk = []
        current_chunk = []

        for token, tag in tagged_sent:
            if tag != "O":
                current_chunk.append((token, tag))
            else:
                if current_chunk:  # if the current chunk is not empty
                    continuous_chunk.append(current_chunk)
                    current_chunk = []
        # Flush the final current_chunk into the continuous_chunk, if any.
        if current_chunk:
            continuous_chunk.append(current_chunk)
        return continuous_chunk

    # call to classifier ner function

    named_entities = get_continuous_chunks(classify)
    named_entities1 = get_continuous_chunks(classify1)
    named_entities_str_tag = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entities]
    named_entities_str_tag1 = [(" ".join([token for token, tag in ne]), ne[0][1]) for ne in named_entities1]

    #predicate generalizer

    def predicate_generator1(readme):
        for words1 in '.!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~':
            readme = readme.replace(words1, '')
        words1 = word_tokenize(readme)
        words1 = [lem.lemmatize(word) for word in words1]
        # words1 = [ps.stem(word)for word in words1]
        source = [word.lower() for word in words1]
        verb1 = []
        for word, pos in nltk.pos_tag(source):
            if (pos == 'VB'):
                verb1.append(word)
        verbs = []
        for token in verb1:
            lemma = [lemma for lemma in vn.classids(token)]
            verbs.append(lemma)
        return verbs

    # predicate generalizer

    def predicate_generator2(readme):
        for words1 in '.!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~':
            readme = readme.replace(words1, '')
        words1 = word_tokenize(readme)
        words1 = [lem.lemmatize(word) for word in words1]
        # words1 = [ps.stem(word)for word in words1]
        source = [word.lower() for word in words1]
        verb1 = []
        for word, pos in nltk.pos_tag(source):
            if (pos == 'VB'):
                verb1.append(word)
        verbs1 = []
        for token in verb1:
            lemma = [lemma for lemma in vn.classids(token)]
            verbs1.append(lemma)
        return verbs1

    pre = predicate_generator1(readme)
    pre1 = predicate_generator2(readmee)

    for words1 in '.!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~':
        readme = readme.replace(words1, '')
    words1 = word_tokenize(readme)
    source = [w for w in words1 if w not in stop_words]
    for words2 in '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~':
        readmee = readmee.replace(words2, '')
    words2 = word_tokenize(readmee)
    suspecious = [w for w in words2 if w not in stop_words]

    #synonyms detector

    synonyms = []
    for token in source:
        for syn in wordnet.synsets(token):      #wordnet database
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())

    synonyms1 = []
    for token1 in suspecious:
        for syn in wordnet.synsets(token1):
            for lemma in syn.lemmas():
                synonyms1.append(lemma.name())

    def flatten(alist):
        newlist = []
        for item in alist:
            if isinstance(item, list):
                newlist = newlist + flatten(item)
            else:
                newlist.append(item)
        return newlist

    vg = flatten(pre)
    vg1 = flatten(pre1)

    #formula for similarity calculation

    def jaccard_distance(a, b):
        """Calculate the jaccard distance between sets A and B"""
        a = set(a)
        b = set(b)
        try:
                # suppose that number2 is a float
                return 1.0 * len(a & b) / min(map(len, (a, b)))
        except ZeroDivisionError:
            return 0.0
        print(a)
        print(b)
        #return 1.0 * len(a & b) / min(map(len, (a, b)))

    kw = "{}".format(jaccard_distance(trigrams1, trigrams))
    dr = "{}".format(jaccard_distance(parse1, parse))
    sd = "{}".format(jaccard_distance(synonyms1, synonyms))
    ner = "{}".format(jaccard_distance(named_entities_str_tag, named_entities_str_tag1))
    pg = "{}".format(jaccard_distance(vg1, vg))

    kw = float(kw)
    dr = float(dr)
    sd = float(sd)
    ner = float(ner)
    pg = float(pg)

    if kw >= 0.1:
        kwv = 10
    elif kw >= 0.05:
        kwv = 5
    elif kw >= 0.03:
        kwv = 3
    else:
        kwv = 0

    if dr >= 0.2:
        drv = 15
    elif dr >= 0.15:
        drv = 13
    elif dr >= 0.1:
        drv = 10
    elif dr >= 0.05:
        drv = 5
    else:
        drv = 0

    if sd >= 0.5:
        sdv = 30
    elif sd >= 0.4:
        sdv = 20
    elif sd >= 0.3:
        sdv = 15
    elif sd >= 0.2:
        sdv = 10
    elif sd >= 0.1:
        sdv = 5
    elif sd >= 0.05:
        sdv = 3
    else:
        sdv = 0

    if ner >= 0.5:
        nerv = 20
    elif ner >= 0.4:
        nerv = 17
    elif ner >= 0.3:
        nerv = 14
    elif ner >= 0.2:
        nerv = 10
    elif ner >= 0.1:
        nerv = 5
    elif ner >= 0.05:
        nerv = 3
    else:
        nerv = 0

    if pg >= 0.5:
        pgv = 25
    elif pg >= 0.4:
        pgv = 20
    elif pg >= 0.3:
        pgv = 15
    elif pg >= 0.2:
        pgv = 10
    elif pg >= 0.1:
        pgv = 5
    elif pg >= 0.05:
        pgv = 3
    else:
        pgv = 0

    result = kwv + drv + sdv + nerv + pgv
    plag = result
    clean = 100 - plag
    objects = ('plagiarized', 'Clean')
    width = 0.20
    yinterval = 10
    pylab.ylim([0, 100])
    performance = [plag, clean]
    y_pos = np.arange(len(performance))
    performance = [plag, clean]
    plt.bar(y_pos, performance, align='edge', width=width, alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.yticks(range(0, max(performance), yinterval))
    plt.title('Plagiarized vs Clean')
    plt.ylabel('Percentage')
    pylab.grid(True)
    res = plt.show()
    return res







    '''
    kw = kw.encode("utf-8")
    dr = dr.encode("utf-8")
    sd = sd.encode("utf-8")
    ner = ner.encode("utf-8")
    pg = pg.encode("utf-8")
    kw = float(kw)
    dr = float(dr)
    sd = float(sd)
    ner = float(ner)
    pg = float(pg)

    if kw >= 0.1:
        kw = "plagiarized" + ' ' + str(kw)
    else:
        kw = "clean" + ' ' + str(kw)

    if dr >= 0.20:
        dr = "plagiarized" + "  " + str(dr)
    else:
        dr = "clean" + "  " + str(dr)

    if sd >= 0.50:
        sd = "plagiarized" + "  " + str(sd)
    else:
        sd = "clean" + "  " + str(sd)

    if ner >= 0.50:
        ner = "plagiarized" + "  " + str(ner)
    else:
        ner = "clean" + "  " + str(ner)
        

    if pg >= 0.50:
        pg = "plagiarized" + "  " + str(pg)
    else:
        pg = "clean" + "  " + str(pg)
     '''



'''   
#parse1 = flatten(parse)


trigrams=list(ngrams(parse1,1))


tree = nltk.Tree.fromstring(file2, read_leaf=lambda x: x.split("/")[0])
tree = tree.leaves()

str1 = ' '.join(tree)
str1 = sent_tokenize(str1)
print(str1)


dep_parser=StanfordDependencyParser(os.environ['STANFORD_PARSER'],os.environ['STANFORD_MODELS'],model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
#parse = dep_parser.raw_parse_sents(str1)
#parse = [list(parse) for parse in dep_parser.raw_parse_sents(str1)]
#parse = sum([[parse.tree() for parse in dep_graphs] for dep_graphs in dep_parser.raw_parse_sents((str1))], [])
#parse = flatten(parse)
parse = sum([[list(parse.triples()) for parse in dep_graphs] for dep_graphs in dep_parser.tagged_parse_sents(((tree),))],[])
print(parse)
parse1 = sum([[list(parse.triples()) for parse in dep_graphs] for dep_graphs in dep_parser.tagged_parse_sents(((tree1),))],[])
print(parse1)


   # print(trigrams1)
'''
