## Self-Attention Mechanism

The self-attention mechanism is a way for each position in a sequence of tokens - such as words in a sentence - to decide how much it should focus on every other position. 
Rather than processing a sequence in a strictly left-to-right or right-to-left manner, self-attention scans the entire set of tokens simultaneously, allowing the model to learn long-range dependencies and contextual relationships more flexibly.

In practical terms, each token gets transformed into three separate vectors: a query, a key, and a value. 
A token’s query vector asks, “What am I looking for?” while a key vector answers, “What information do I contain?”, and a value vector provides the actual content. 
By taking the dot product of queries and keys, and then normalizing these scores, each token can “attend” to the relevant parts of every other token’s value. 
This yields a weighted sum of the surrounding information, creating richer token representations that incorporate context from the entire sequence.

In the provided code, the steps are simple yet illustrative: you embed each token into a dense vector, project those embeddings into query, key, and value spaces, compute attention scores using a dot product (scaled and softmaxed for stability), and then aggregate the value vectors. 
This final step produces updated representations for each token that capture how they interact and influence each other within the sentence.


## Masked Self-Attention Mechanism

Masked self-attention is a variation of the self-attention mechanism where certain positions in the input sequence are prevented from attending to future positions. 
This is particularly useful in autoregressive tasks such as language modeling or text generation, where predictions must be made sequentially without access to future tokens. 
The masking ensures that at each step, a token can only attend to itself and previous tokens, preserving causality in the model’s predictions. 
This is typically achieved by applying a mask—a matrix that assigns very low values (e.g., negative infinity) to disallowed positions before applying the softmax operation, effectively setting their attention weights to zero.


## Multi-Head Self-Attention Mechanism

Multi-head attention extends the self-attention mechanism by running multiple “heads” in parallel, each one with its own set of query, key, and value projection matrices. 
The idea is that each head can learn to focus on different aspects of the sequence - perhaps one head attends to immediate neighbors, while another focuses on longer-range dependencies or certain semantic traits.

After each head computes its own attention output, these multiple representations are concatenated and transformed once more to produce the final output. 
This setup helps the model capture a richer variety of patterns and relationships than a single attention mechanism could manage on its own, often leading to more robust and nuanced representations of sequence data.
