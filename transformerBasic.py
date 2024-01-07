import numpy as np
import math

# Find vocabSize
input = "I drink and I know things\nWhen you play the game of thrones you win or you die\nThe true enemy won't wait out the storm, He brings the storm"
N = input.split()
vocabList = list(set(N))

# Encode vocab
mapVocab = list(enumerate(vocabList))

# Calculate embedding
# Determine input sentence
#sentenceIn =

# Determine length of input sentence
#senLen = len(sentenceIn)
senLen = 6
d = 10
dk = d
dModel = d
# Generate embedding array from input sentence (token x token) matrix of random numbers (0-1)
inputEmbedding = np.random.rand(d,senLen)
posEmbedding = np.zeros((d,senLen))

# Calculate positional embedding
# Find number of tokens
tokens = 6

# For each token
for pos in range(tokens):
    # For each position
    for i in range(d):
        if i%2 == 0:
            peEven = math.sin(pos/10000**((2*i)/dModel))
            posEmbedding[i][pos] = peEven
        else:
            peOdd = math.cos(pos/10000**((2*i)/dModel))
            posEmbedding[i][pos] = peOdd
encoderInput = np.add(posEmbedding,inputEmbedding)

# Calculate the QKV query matrices size(d,4 (this can be any number))
qWeights = np.random.rand(senLen,4)
kWeights = np.random.rand(senLen,4)
vWeights = np.random.rand(senLen,4)

qQuery = np.matmul(encoderInput,qWeights)
kQuery = np.matmul(encoderInput,kWeights)
vQuery = np.matmul(encoderInput,vWeights)

QKt_sqrt_d = (np.matmul(qQuery,kQuery.transpose()))/(math.sqrt(dk))

attention = np.matmul((np.exp(QKt_sqrt_d)/sum(np.exp(QKt_sqrt_d))),vQuery)




""" 
# Attention formula
attention(Q, K, V) = softmax((QK')/(sqrt(dk)))

# head 
head(i) = attention(QW(iQ), KW(iK), VW(iV))

resolution.org.uk -> find a law professional -> advanced search
 """