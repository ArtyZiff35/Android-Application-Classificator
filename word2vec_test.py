from gensim.models import KeyedVectors
from scipy import spatial

filename = './word2vecModels/GoogleNews-vectors-negative300.bin'
print('Loading model...')
model = KeyedVectors.load_word2vec_format(filename, binary=True)
print(model['queen'])
print(model['queen'][299])



print("\n\n")

result = 1 - spatial.distance.cosine((model['queen']+model['man']), model['king']) # Cosine similarity (1-distance)
print("\n")
print(result)

result = 1 - spatial.distance.cosine(model['queen'], model['girl']) # Cosine similarity (1-distance)
print("\n")
print(result)

result = 1 - spatial.distance.cosine(model['queen'], model['woman']) # Cosine similarity (1-distance)
print("\n")
print(result)