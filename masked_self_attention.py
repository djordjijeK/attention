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

# Generate a causal mask (look-ahead mask)
seq_len = attentions.shape[0]  # number of tokens
mask = torch.tril(torch.ones(seq_len, seq_len))  # lower-triangular

# Convert 1/0 mask into large negative numbers for "future" positions.
# mask == 1 => keep the score as is
# mask == 0 => set the score to -1e9
mask = mask.masked_fill(mask == 0, float('-1e9'))

# Apply the mask before the softmax
# For valid positions, adding 0 means no change; for invalid positions, adding -1e9 leads to ~0 after softmax
attentions_masked = attentions + mask

# Scale and normalize
scaled_dot_product_attention = F.softmax(attentions_masked / (d_key**0.5), dim=1)

# Context vectors
context_vectors = scaled_dot_product_attention @ values


if __name__ == '__main__':
    print(f"Input sequence: {input_sequence}")
    print(f"Vocabulary: {vocabulary}")
    print(f"Input sequence encoded (sequence of tokens): {input_sequence_encoded}")
    print("-----------------------------------------------------------------------------------")
    print("Input sequence embeddings (4-dimensional per token):")
    print(embedded_input_sequence)
    print("-----------------------------------------------------------------------------------")
    print("Queries:")
    print(queries)
    print("Keys:")
    print(keys)
    print("Values:")
    print(values)
    print("-----------------------------------------------------------------------------------")
    print("Raw attentions:")
    print(attentions)
    print("-----------------------------------------------------------------------------------")
    print("Causal mask:")
    print(mask)
    print("-----------------------------------------------------------------------------------")
    print("Masked attentions (with causal mask applied):")
    print(attentions_masked)
    print("-----------------------------------------------------------------------------------")
    print("Scaled attentions (after softmax):")
    print(scaled_dot_product_attention)
    print("-----------------------------------------------------------------------------------")
    print("Context vectors:")
    print(context_vectors)