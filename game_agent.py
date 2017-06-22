"""
Players and the score functions that guide them for the board game Isolation.

Udacity: Artificial Intelligence Nanodegree
Project 2: Isolation, with knight-based movement
Author: Hamilton Hoxie Ackerman, with starting code provided by Udacity
"""


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


### Features of the board used in linear combos for my custom_score functions
def num_moves(game, player):
    """How many legal moves does `player` have in `game`?"""
    return len(game.get_legal_moves(player))

def num_moves_opponent(game, player):
    """How many legal moves does the opponent of `player` have in `game`?"""
    return len(game.get_legal_moves(game.get_opponent(player)))

def dist_from_center(game, player):
    """What is `player`'s Manhattan distance from the center of the board?"""
    center_x = game.height / 2
    center_y = game.width / 2
    player_x, player_y = game.get_player_location(player)
    return abs(center_x - player_x) + abs(center_y - player_y)

HEURISTICS = (num_moves, num_moves_opponent, dist_from_center)


### Combine these features into score functions
def make_custom_score(weights):
    """
    Create a score function that's a linear combination of the functions in
    `heuristics`, where the coefficients come from `weights` as follows:
    """

    assert len(weights) == len(HEURISTICS)
    def score_fn(game, player):
        """
        Score function for an Isolation player.

        Scores the `game` board from the perspective of `player` using the following
        weighted combination of heuristic functions:

        {}
        """.format("\n".join(["{}: {}".format(fn.__name__, round(w, 3))
                              for fn, w in zip(HEURISTICS, weights)]))

        if game.is_loser(player):
            return float("-inf")
        if game.is_winner(player):
            return float("inf")
        return sum(w * fn(game, player) for w, fn in zip(weights, HEURISTICS))
    return score_fn


WEIGHTS = [
    (1.58, -1.3, -1.81),
    (4.08, -3.3, 0.89),
    (-1.22, -2.52, 3.56)
]

CUSTOM_SCORES = [make_custom_score(ws) for ws in WEIGHTS]


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return CUSTOM_SCORES[0](game, player)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return CUSTOM_SCORES[1](game, player)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return CUSTOM_SCORES[2](game, player)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate successors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        legal_moves = game.get_legal_moves(self)
        if not legal_moves:
            return (-1, -1)
        best_move = legal_moves[0]

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    def minimax(self, game, depth):
        """
        Implement depth-limited minimax search algorithm as described in the
        lectures.

        Based on
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves(self)
        if not legal_moves:
            return (-1, -1)

        best_move = legal_moves[0]
        best_v = float("-inf")
        for move in legal_moves:
            new_game = game.forecast_move(move)
            new_v = self.min_value(new_game, depth - 1)
            if new_v > best_v:
                best_v, best_move = new_v, move
        return best_move


    def max_value(self, game, num_plies):
        """
        minimax helper function, part of mutually recursive pair max_value/min_value

        Based on
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        Parameters
        ----------
        game : isolation.Board
            Represents the current game state

        num_plies : int
            The maximum number of plies to search in the game tree (after this
            one) before aborting. If num_plies is 0, then this search has
            reached its maximum depth, so our best guess for a value is our
            heuristic's evaluation of the current state. Otherwise, consider
            all legal moves, and the opponent's strongest reactions to each of
            them, via min_value.

        Returns
        -------
        float
            An estimate of the max value that can be achieved by a legal move,
            assuming our opponent is trying to minimize our value
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Is the game over? Is the search over?
        legal_moves = game.get_legal_moves()
        if not legal_moves or num_plies == 0:
            return self.score(game, self)

        # Find the best move, assuming opponent will minimize our value, in
        # the remaining (num_plies - 1) plies of search
        v = float("-inf")
        for move in legal_moves:
            new_game = game.forecast_move(move)
            v = max(v, self.min_value(new_game, num_plies - 1))
        return v


    def min_value(self, game, num_plies):
        """
        minimax helper function, part of mutually recursive pair max_value/min_value

        Based on
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        Parameters
        ----------
        game : isolation.Board
            Represents the current game state

        num_plies : int
            The maximum number of plies to search in the game tree (after this
            one) before aborting. If num_plies is 0, then this search has
            reached its maximum depth, so our best guess for a value is our
            heuristic's evaluation of the current state. Otherwise, consider
            all legal moves, and our strongest response to each of them, via
            max_value.

        Returns
        -------
        float
            An estimate of the min value that can be achieved by a legal move,
            assuming the opponent will maximize value in whatever configuration
            it faces
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Is the game over? Is the search over?
        legal_moves = game.get_legal_moves()
        if not legal_moves or num_plies == 0:
            return self.score(game, self)

        # Find the opponent's best move, i.e. the one that minimizes our
        # estimated value, in the remaining (num_plies - 1) plies of search
        v = float("inf")
        for move in legal_moves:
            new_game = game.forecast_move(move)
            v = min(v, self.max_value(new_game, num_plies - 1))
        return v


class AlphaBetaPlayer(IsolationPlayer):
    """
    Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        legal_moves = game.get_legal_moves(self)
        if not legal_moves:
            return (-1, -1)

        best_move = legal_moves[0]
        max_possible_moves = game.width * game.height
        for max_depth in range(1, max_possible_moves + 1):
            try:
                best_move = self.alphabeta(game, max_depth)
            except SearchTimeout:
                break  # Handle any actions required after timeout

        # Return the best move from the last completed search iteration
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """
        Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        A modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        """

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves(self)
        if not legal_moves:
            return (-1, -1)

        best_move = legal_moves[0]
        best_v = float("-inf")

        for move in legal_moves:
            new_game = game.forecast_move(move)
            v = self.min_value(new_game, alpha, beta, depth-1)
            if v > best_v:
                best_v = v
                best_move = move
            alpha = max(alpha, v)
        return best_move


    def max_value(self, game, alpha, beta, num_plies):
        """
        alphabeta helper function, part of mutually recursive pair max_value/min_value

        Based on
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        Parameters
        ----------
        game : isolation.Board
            Represents the current game state

        alpha: float
            The value of the best (highest-value) choice found so far at any
            choice point along the search path for the maximum attainable value

        beta: float
            The value of the best (lowest-value) choice found so far at any
            choice point along the search path for the minimum attainable value

        num_plies : int
            The maximum number of plies to search in the game tree (after this
            one) before aborting. If num_plies is 0, then this search has
            reached its maximum depth, so our best guess for a value is our
            heuristic's evaluation of the current state. Otherwise, consider
            all legal moves, and the opponent's strongest reactions to each of
            them, via min_value.

        Returns
        -------
        float
            An estimate of the max value that can be achieved by a legal move,
            assuming our opponent is trying to minimize our value
        """
        # Are we out of time?
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Is the game over? Is the search over?
        legal_moves = game.get_legal_moves()
        if not legal_moves or num_plies == 0:
            return self.score(game, self)

        # Find the best move, assuming opponent will minimize our value, in
        # remaining (num_plies - 1) plies of search
        v = float("-inf")
        for move in legal_moves:
            new_game = game.forecast_move(move)
            v = max(v, self.min_value(new_game, alpha, beta, num_plies - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v


    def min_value(self, game, alpha, beta, num_plies):
        """
        alphabeta helper function, part of mutually recursive pair max_value/min_value

        Based on
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        Parameters
        ----------
        game : isolation.Board
            Represents the current game state

        alpha: float
            The value of the best (highest-value) choice found so far at any
            choice point along the search path for the maximum attainable value

        beta: float
            The value of the best (lowest-value) choice found so far at any
            choice point along the search path for the minimum attainable value

        num_plies : int
            The maximum number of plies to search in the game tree (after this
            one) before aborting. If num_plies is 0, then this search has
            reached its maximum depth, so our best guess for a value is our
            heuristic's evaluation of the current state. Otherwise, consider
            all legal moves, and our strongest response to each of them, via
            max_value.

        Returns
        -------
        float
            An estimate of the min value that can be achieved by a legal move,
            assuming the opponent will maximize value in whatever configuration
            it faces
        """

        # Are we out of time?
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Is the game over? Is the search over?
        legal_moves = game.get_legal_moves()
        if not legal_moves or num_plies == 0:
            return self.score(game, self)

        # Find the opponent's best move, i.e. the one that minimizes our
        # estimated value, in the remaining (num_plies - 1) plies of search
        v = float("inf")
        for move in legal_moves:
            new_game = game.forecast_move(move)
            v = min(v, self.max_value(new_game, alpha, beta, num_plies - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
