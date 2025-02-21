import torch

from torch.functional import F
from torch.nn import Embedding, Parameter


embedding_size = 4  # each token is represented as a 16-dimensional vector

d_key = 8           # each key is represented as a 32-dimensional vector
d_query = 8         # each query is represented as a 32-dimensional vector
d_value = 16        # each value is represented as a 64-dimensional vector


# input sequence for the self-attention layer
input_sequence = 'The sky is blue'

# tokenize input
words = input_sequence.replace(',', '').split()

# assign a unique integer index to each token
vocabulary = {word: index for index, word in enumerate(sorted(words))}

# encode words into integer indices based on the vocabulary
input_sequence_encoded = torch.tensor([vocabulary[word] for word in words])

# convert integer-encoded tokens into dense vector embeddings
embeddings = Embedding(len(vocabulary), embedding_size)
embedded_input_sequence = embeddings(input_sequence_encoded)

# projection matrices to transform embeddings into query, key, and value vectors
W_query = Parameter(torch.rand(d_query, embedding_size))   # query projection matrix
W_key = Parameter(torch.rand(d_key, embedding_size))       # key projection matrix
W_value = Parameter(torch.rand(d_value, embedding_size))   # value projection matrix

# compute query, key, and value vectors
queries = embedded_input_sequence @ W_query.T  # Shape: (num_tokens, d_query)
keys = embedded_input_sequence @ W_key.T       # Shape: (num_tokens, d_key)
values = embedded_input_sequence @ W_value.T   # Shape: (num_tokens, d_value)

# compute raw attention scores (dot product between queries and keys)
attentions = queries @ keys.T  # Shape: (num_tokens, num_tokens)

# apply scaling to stabilize gradients and normalize attention scores
scaled_dot_product_attention = F.softmax(attentions / d_key**0.5, dim=1)  # Shape: (num_tokens, num_tokens)

# compute context vectors as a weighted sum of values using attention scores
context_vectors = scaled_dot_product_attention @ values  # Shape: (num_tokens, d_value)


if __name__ == '__main__':
    print(f"Input sequence: {input_sequence}")
    print(f"Vocabulary: {vocabulary}")
    print(f"Input sequence encoded (sequence of tokens): {input_sequence_encoded}")
    print("-----------------------------------------------------------------------------------")
    print("Input sequence embeddings: each token is represented as a 4-dimensional vector")
    print(embedded_input_sequence)
    print("-----------------------------------------------------------------------------------")
    print(f"Queries (embeddings projected as queries): {queries.shape}")
    print(queries)
    print(f"Keys (embeddings projected as keys): {keys.shape}")
    print(queries)
    print(f"Values (embeddings projected as values): {values.shape}")
    print(values)
    print("-----------------------------------------------------------------------------------")
    print(f"Raw attentions (how much each token 'attend' to every other token): {attentions.shape}")
    print(attentions)
    print(f"Scaled attentions: {scaled_dot_product_attention.shape}")
    print(scaled_dot_product_attention)
    print("-----------------------------------------------------------------------------------")
    print(f"Context vectors: {context_vectors.shape}")
    print(context_vectors)