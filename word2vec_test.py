from gensim.models import KeyedVectors

filename = './word2vecModels/GoogleNews-vectors-negative300.bin'
print('Loading model...')
model = KeyedVectors.load_word2vec_format(filename, binary=True)
print(model['queen'])
print(model['queen']+model['king'])
