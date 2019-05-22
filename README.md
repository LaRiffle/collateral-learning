# Collateral Learning

## Motivation

Imagine that you train a neural network to perform a specific task, and you discover it has also learned information which makes it possible to perform another completely different task, which is very sensitive. Is this possible? What can you do to prevent this?

#### Related but distinct notions
 > **Transfer learning**: You train a model to perform a specific task, and you reuse this pre-trained model to perform a related task on possibly different data. _The difference in collateral learning is that the second task is of a different nature, while the data used should be closely related to the original one_.
 
 > **Adversarial learning**: You corrupt the input data to fool the model and reduce its performance. _The difference in collateral learning is that you don't modify the input but you try to disclose hidden sensitive information about it using the model output_.

## Context

Let's assume you have a semi-private **trained** network performing a prediction task of some nature `pred1`. This means you have a network with the first layers encrypted and the last ones visible in clear. The structure of the network could be written like this: `x -> Private -> output(x) -> Public -> pred1(x)`. For example, `pred1(x)` could be the age based on an face picture input `x`, or the text translation of some speech record.

There are several encryption schemes (among which MPC, FHE, SNN, etc.) but we here interested in Functional Encryption (FE) which can be used to encrypt quadratic networks as it is done [here](https://eprint.iacr.org/2018/206), where actually you are even given `x` encrypted (i.e. `Enc(x) -> Private -> output(x) ...`). One reason for this setting with two phases is that encryption is very expensive or restrictive (like FE as of today which only support a single quadratic operation), but you can improve the model accuracy by adding a neural network in clear which will leverage the encrypted part output.

On the testing phase, if you make a prediction on an input, one can only observe the neuron activations starting from the output of the private part, `output(x)` with our notations. Hence, `output(x)` is _the best you can know_ about the original input.

We investigate here how an adversary could query this trained model `x -> Private -> output(x) -> Public -> pred1(x)` with an item `x` for which it's aware it has another feature `pred2(x)` that can be classified. If for example the input `x` is a face picture then `pred2(x)` could be the gender for example, or it could be the origin of the person talking if `x` is a speech record. The goal of the adversary is to learn another network based on `output(x)` which can perform a prediction task `output(x) -> NN -> pred2(x)` for encrypted items `x`.

## Use case

So what's the use case? For example, imagine you provide a secure service to write down speech records and people give you encrypted speech records. All you can read and exploit in clear is `output(x)` which can be a relatively short vector compared to `x`, which is enough for you to detect the text and deliver your service. The question is: Can you use `output(x)` to detect the ethnic origin of the person speaking?

## Our approach

We give concrete examples of this in our repository to answer this question. Even if some existing datasets exist that are suitable for two distinct and "relatively" independent learning tasks, like the face dataset [imdb-wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), we have proposed a 60.000 items artificial letter character dataset inspired from MNIST, where several fonts were used to draw the characters to which deformation is added. Hence, we can ensure complete independence between the two features to classify and adjust the difficulty of classification to the current capabilities of Functional Encryption.

![Bilby Stampede](./img/collateral_learning.png)

Our work is detailed in the tutorials section. Any comments are welcome!

If you're enthusiastic about our project, ⭐️ it to show your support! :heart:
