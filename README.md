# MARKOV SPEECH GENERATION

In this project, I create a simple Markov chain model and use it to generate sentences from our favorite green Jedi. This notebook contains a brief explanation of my model 
and demonstrates how to deploy it.

### MODEL OVERVIEW

A **Markov chain** is formally defined as a sequence of random variables $X_1, X_2, ..., X_n$ such that the following relation holds for all $n$:

$$
\begin{align}
    P(X_n | X_{n-1}, X_{n-2}, ..., X_1) = P(X_n | X_{n-1})
\end{align}
$$

In easier, less symbolic terms, a Markov chain is a way of modeling a special (but common) class of random systems in the world in which the state of the system is only dependent on the previous state (and no other former states).

With any stochastic process, it is typical to identify a set of possible states, often denoted $S$, and a collection of transition probabilities $p_S$ which define the likelihood of transitioning from one state into another. So assuming a finite number of states, if we are given a state $S_i$, there exists some vector $p_{S_i}$ with associated transition probabilities into every other state (including sometimes, the same $S_i$).

Here is a concrete example. Suppose the weather can be modeled as a Markov chain. For the sake of simplicity, suppose also that the set of possible states is $\{sun, rain\}$ and the transition probabilities $p_S$ are as follows:

$$
\begin{align*}
    p(sun|rain) = 0.3 \\
    p(rain|rain) = 0.7 \\ 
    p(sun|sun) = 0.6 \\
    p(rain|sun) = 0.4 \\
\end{align*}
$$

A common way to efficiently model a system like this is to represent it as a matrix $A$ of transition probabilities, where each column corresponds to the current state (sun or rain today) and each row corresponds to the possible next state (sun or rain tomorrow). In this case, we would have

$$
A =
\begin{bmatrix}
0.6 & 0.3 \\
0.4 & 0.7
\end{bmatrix}
$$

To simulate this system, then, amounts to determining what the current state is and making a random draw from a categorical distribution defined by that state's corresponding transition probabilities (indicated by the matrix's column associated with the current state). The result of the draw is the next state.

This process can be repeated multiple times, even indefinitely, until a terminating state (where the probability of transitioning into any other state is 0) is reached. With sentence generation, that terminating state is the period at the end of the sentence. In other applications, a terminating state could be vastly different.

Let's illustrate this process with our weather example, expanding it slightly to include the miniscule chance of a **catastrophic meteor storm**, which is our terminating state. We define new transition probabilities:

$$
\begin{align*}
    p(sun|rain) = 0.29 \\
    p(rain|rain) = 0.70 \\
    p(meteor storm | rain) = 0.01 \\
    p(sun|sun) = 0.60 \\
    p(rain|sun) = 0.38 \\
    p(meteor storm | sun) = 0.02 \\
    p(sun | meteor storm) = 0.00 \\
    p(rain | meteor storm) = 0.00 \\
    p(meteor storm | meteor storm) = 1.00
\end{align*}
$$

with corresponding transition matrix

$$
A =
\begin{bmatrix}
0.60 & 0.29 & 0.00 \\
0.38 & 0.70 & 0.00 \\
0.02 & 0.01 & 1.00
\end{bmatrix}
$$

If we started with an initial state of **rain** and simulated indefinitely, our Markov chain would spit out a forecast resembling

$ [rain, rain, rain, sun, sun, sun, sun, sun, rain, ... , sun, meteor storm] $

after which we would terminate (since we cannot transition out of the catastrophic meteor storm state).

### MARKOV SENTENCE GENERATION

To produce sentences that imitate the style of a specific person, we first require training data -- sentences actually spoken by that person in the past. From this training data, the transition matrix can be iteratively constructed. Simply assign each word used by the speaker to a column of a zeros matrix and increment entries where we "transitioned" from one word to another. Then, normalize the columns to form probability vectors. 

It is also useful to add two artificial states, $tart and $top, to the beginning and end of each training sentence respectively. This ensures that as we create a Markov forecast, we actually form a sentence instead of an endless stream of words. The artificial states can be excluded from the final forecast.

### CODE DEMO

Before we get going with our fake Yoda, we need to import the necessary classes.

```python
from src.models.sentence_generator import SentenceGenerator
```

Next, we create a SentenceGenerator object and load in our training corpus (a text file with all of Yoda's lines in the Star Wars saga). 

*Note that the code as currently designed requires the training data to be formatted as one sentence per line. Future functionality will expand the flexibility of my current setup.*

```python
fake_yoda = SentenceGenerator('assets/yoda.txt')
```


All that is left to do now is call the `babble` method of our fake yoda, and watch the results! Let's generate 10 random sentences from Yoda and see how he does:

```python
for _ in range(10):
    print(fake_yoda.babble())
```

```
Strong am wondering - like his father.
Easily they flow quick to eat as his life has he is.
A Jedi uses the burden were you!
At an apprentice.
Use your Padawan learner, I do you?
With the future is your destiny; consume you not.
For eight hundred years have I am sure of: do not!
Too much pride in this be?
I would say about it, I with the Council is why are calm, at peace.
Much anger there is, Commander.
```

Ok, so maybe we still have some work to do before we convince anyone that this is actually Yoda speaking. But this is a good start! Hopefully you've been able to see the power of Markov chains through this short demo.