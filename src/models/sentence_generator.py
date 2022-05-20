import numpy as np
from markov_chain import MarkovChain

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        transition_matrix ((n,n), ndarray): the column-stochastic transition
            matrix for the provided English corpus.
        state_labels (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.
        states_index_map(dict(str, int)): a dictionary mapping each label
            from the states list to its corresponding column index
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. We assume that the file has one complete sentence
        written on each line.
        """
        sentences = open(filename, 'r').readlines()
        unique_words = set()

        for sentence in sentences:
            for word in sentence.strip().split():
                unique_words.add(word)

        self.state_labels = [word for word in unique_words]

        # Add labels "$tart" and "$top" to the set of states labels.
        self.state_labels.insert(0, "$tart")
        self.state_labels.append("$top")

        # Build transition matrix.
        n = len(self.state_labels)
        self.transition_matrix = np.zeros(shape=(n,n))
        self.states_index_map = { state:index for state, index in zip(self.state_labels, range(n)) }

        for sentence in sentences:

            # Split the sentence into a list of words.
            sentence_words = sentence.strip().split()

            # Prepend "$tart" and append "$top" to the list of words.
            sentence_words.insert(0, "$tart")
            sentence_words.append("$top")

            for i in range(len(sentence_words) - 1):

                # Save current state and next state.
                current_word = sentence_words[i]
                next_word = sentence_words[i+1]

                # Get corresponding index of current and next state.
                current_state_index = self.states_index_map[current_word]
                next_state_index = self.states_index_map[next_word]

                # Increment the (i,j) entry to mark a transition from j to i.
                self.transition_matrix[next_state_index, current_state_index] += 1
                    
        # Make sure the $top state transitions to itself.
        self.transition_matrix[n-1,n-1] = 1

        # Normalize each column.
        for i in range(n):
            self.transition_matrix[:,i] /= self.transition_matrix[:,i].sum()

    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.
        """
        markov_chain = MarkovChain(self.transition_matrix, self.state_labels)
        sentence = markov_chain.path('$tart', '$top')
        sentence.remove('$tart')
        sentence.remove('$top')

        return ' '.join(sentence).strip()