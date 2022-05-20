import numpy as np

class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        transition_matrix ((n,n), ndarray): the column-stochastic transition
            matrix for a Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.
        states_index_map(dict(str, int)): a dictionary mapping each label
            from the states list to its corresponding column index
    """
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        m, n = A.shape

        if not np.allclose(A.sum(axis=0), np.ones(n)):
            raise ValueError("Provided transition matrix is not column stochastic.")

        self.transition_matrix = A

        # Initialize list of possible states
        if states is None:
            self.states = [i for i in range(n)]
        else:
            self.states = states

        # Initialize state labels index map
        self.states_index_map = { state:index for state, index in zip(self.states, range(n)) }
        

    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transition to.
        """
        current_state = self.states_index_map[state]
        transition_probabilities = self.transition_matrix[:,current_state]
        next_state = np.argmax(np.random.multinomial(n=1, pvals=transition_probabilities))

        # Return corresponding label for next state.
        return self.states[next_state]


    def walk(self, start, N):
        """Starting at the specified state, transition from state to state N-1 times, 
        recording the state label at each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        visited_states = []

        current_state = start
        visited_states.append(current_state)

        for i in range(N-1):
            current_state = self.transition(state=current_state)
            visited_states.append(current_state)

        return visited_states


    def path(self, start, stop):
        """Beginning at the initial state, transition from state to state until
        arriving at the terminating state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        visited_states = []

        current_state = start
        visited_states.append(current_state)

        while current_state != stop:
            current_state = self.transition(state=current_state)
            visited_states.append(current_state)

        return visited_states