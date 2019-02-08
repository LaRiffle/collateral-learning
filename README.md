# Collateral Learning

## Motivation

Imagine that you train a neural network to perform a specific task, and you discover it has also learned information which makes it possible to perform another completely different task, which is very sensitive. Is this possible? What can you do to prevent this?

#### Related but distinct notions
 > **Transfer learning**: You train a model to perform a specific task, and you reuse this pre-trained model to perform a related task on possibly different data. _The difference with collateral learning is that the second task is of a different nature, while the data used should be closely related to the original one_.
 
 > **Adversarial learning**: You corrupt the input data to fool the model and reduce its performance. _The difference with collateral learning is that you don't modify the input but you try to diclose hidden sensitive information about it using the model ouput which is supposed to be un-sensitive_.

## Context

Let's assume you have a semi-private **trained** network performing a prediction task of some nature `pred1`, ie a network with the first layers encrypted and the last ones visible in clear. The pipeline of the network could be written like this: `x -> Private -> output(x) -> Public -> pred1(x)`. `pred1(x)` could be the age based on an face picture input `x`, or the text traduction of some speech record.

There are several encryption schemes (among which MPC, FHE, SNN, etc.) but we here interested in Functional Encryption (FE) which can be used to encrypt quadratic networks as it is done [here](https://eprint.iacr.org/2018/206), where actually you are even given `x` encrypted (i.e. `Enc(x) -> Private -> output(x) ...`). One reason for this setting with two phases is that encryption is very expensive or restrictive (like FE as of today), but you can add a neural network in clear to leverage the encrypted part output and improve the model accuracy.

On testing phase, if you make a prediction on an input, one can only observe the neuron activations starting from the output of the private part, `output(x)`. Hence, `output(x)` is _the best you can know_ about the original input.

We investigate here how an adversary could query this trained model `x -> Private -> output(x) -> Public -> pred1(x)` with `x` items for which it's aware of another feature `pred2(x)`, _that the trained model is not supposed to be aware of_. If for example the input `x` is a face picture then `pred2(x)` could be the gender for example, or it could be the origin of the person talking if `x` is a speech record. The goal of the adversary is to learn another network which can perform a prediction task `output(x) -> NN -> pred2(x)` for non-owned and thus non-readable `x` items.

## Use case

So what's the use case? For example, imagine you provide a secure service to write down speech records and people give you encrypted speech records. All you can read and exploit in clear is `output(x)` which can be a relatively short vector compared to `x`, even though it is enough for you to detect the text and deliver your service. The question is: Can you use `output(x)` to detect the ethnic origin of the person speaking?

## Our approach

We aim at giving concrete examples of this in our repository to answer this question. We have searched for datasets which are suitable for two distinct and "relatively" independent tasks. A face dataset such as [this one](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) which gathers images from imdb and wikipedia could be suitable, but as in our context we want to encrypt the first model with Functional Encryption, we need to choose a simpler task. Therefore we have proposed an artificial letter character dataset, vastly inspired from MNIST, where several fonts were used to draw the characters. We have applied some noise and deformation to build a 60.000 item dataset.

Our work is detailed in the notebook section. Any comments are welcome!

If you're enthusiastic about our project, ⭐️ it to show your support! :heart:
