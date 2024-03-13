# decoder-from-scratch
Implementing a decoder-only model like GPT2 from scratch using PyTorch and tiktoken.

## Overview
Code to build a decoder-only transformers model akin to OpenAI's GPT2 using
just PyTorch.  My criteria for success was to be able to train the model on
actual text and have it generate a set of phrases given an input sequence.

I used the Jules Verne novel 'The Mysterious Island' for my training corpus.

## Tokenization
To focus only on programming the transformer architecture, I opted to use
pre-trained tokenizers, an idea I stole from [Andrej Karpathy](https://github.com/karpathy/minbpe).

## Sequence structuring
It took me some time to understand how to deal with stop words and end-of-sequence
tokens.  Since I'm training on just one piece of text, I decided to break up
the text into sliding window sequences of 40 tokens.  Since I discarded the last
incomplete sequence, I didn't actually have any stop words.

As for the EOS, when a company is training a new language model, they will of
course have more than 1 document, and end-of-sequence tokens would be relevant.
In my case, however, there is just 1 end of sequence, so ignoring it is fine.

The implication is that for inference, we do not expect the model to generate
a token for when the sequence should end.

## Model parameters
I used a batch size of 50, a feed-forward network dimension of 20 with 10 decoder blocks.
I started using a small d_model (aka hidden size aka embedding size) - around 10.
However, the loss across epochs during training was not improving.  Considering
that GPT2 was trained with a d_model of 768, I increased it to 500 and
10 attention heads.

## Infastructure used
I attempted to train locally on my CPU but it was taking a very long time (~2
hours / epoch).  I switched to a Colab Pro A100 GPU, but quickly exceeded the
allocated quotas, and Google blocked access to GPUs (except for what seemed
to be 20 minutes of use per day).  So I finally switched to a NVidia Tesla T4 on
AzureML, which cost me about $4.80 to train for 10 epochs.

## Inference
The results on some starter sequences with greedy search were repetivie, so I
also implemented a multinomial search strategy. This improved, but unfortunately
the result is still incoherent.

Given that the loss kept decreasing all the way up to epoch 10, my conclusion
is that training the model further would likely improve the predictions.

