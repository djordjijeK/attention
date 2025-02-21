import torch

from torch.functional import F
from torch.nn import Embedding, Parameter


heads = 3           # number of self attention heads
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

# projection matrices to transform embeddings into query, key, and value vectors (now we have heads of them)
multihead_W_query = Parameter(torch.rand(heads, d_query, embedding_size))
multihead_W_key = Parameter(torch.rand(heads, d_key, embedding_size))
multihead_W_values = Parameter(torch.rand(heads, d_value, embedding_size))

# convert integer-encoded tokens into dense vector embeddings
embeddings = Embedding(len(vocabulary), embedding_size)
embedded_input_sequence = embeddings(input_sequence_encoded)

multihead_embedded_input_sequence = embedded_input_sequence.repeat(heads, 1, 1)

# compute query, key, and value vectors (now heads number of times)
multihead_queries = torch.bmm(multihead_embedded_input_sequence, multihead_W_query.transpose(2, 1))
multihead_keys = torch.bmm(multihead_embedded_input_sequence, multihead_W_key.transpose(2, 1))
multihead_values = torch.bmm(multihead_embedded_input_sequence, multihead_W_values.transpose(2, 1))

# compute raw attention scores (dot product between queries and keys in parallel)
attentions = torch.bmm(multihead_queries, multihead_keys.transpose(2, 1))

# apply scaling to stabilize gradients and normalize attention scores
scaled_dot_product_attention = F.softmax(attentions / d_key**0.5, dim=1)  # Shape: (heads, num_tokens, num_tokens)

# compute context vectors as a weighted sum of values using attention scores (in parallel for all heads)
context_vectors = scaled_dot_product_attention @ multihead_values  # Shape: (heads, num_tokens, d_value)


if __name__ == '__main__':
    print(f"Input sequence: {input_sequence}")
    print(f"Vocabulary: {vocabulary}")
    print(f"Input sequence encoded (sequence of tokens): {input_sequence_encoded}")
    print("-----------------------------------------------------------------------------------")
    print(f"Queries (embeddings projected as queries): {multihead_queries.shape}")
    print(f"Keys (embeddings projected as keys): {multihead_keys.shape}")
    print(f"Values (embeddings projected as values): {multihead_values.shape}")
    print("-----------------------------------------------------------------------------------")
    print(f"Scaled attentions: {scaled_dot_product_attention.shape}")
    print("-----------------------------------------------------------------------------------")
    print(f"Context vectors: {context_vectors.shape}")