"""
Estimate the strength rating of a student defined heuristic by competing
against fixed-depth minimax and alpha-beta search agents in a round-robin
tournament.
"""

import itertools
import random

from collections import namedtuple

from pathos.multiprocessing import ProcessingPool as PPool

from isolation import Board
from sample_players import (RandomPlayer, open_move_score,
                            improved_score, center_score)
from game_agent import (MinimaxPlayer, AlphaBetaPlayer, custom_score,
                        custom_score_2, custom_score_3)

NUM_PROCS = 4
NUM_MATCHES = 2  # number of matches against each opponent
TIME_LIMIT = 150  # number of milliseconds before timeout


Agent = namedtuple("Agent", ["player", "name"])


def _run(*args):
    idx, p1, p2, moves = args[0]
    game = Board(p1, p2)
    for m in moves:
        game.apply_move(m)
    winner, _, termination = game.play(time_limit=TIME_LIMIT)
    return (idx, winner == p1), termination


def play_round(cpu_agent, test_agents, win_counts, num_matches):
    """
    Compare the test agents to the cpu agent in "fair" matches.

    "Fair" matches use random starting locations and force the agents to
    play as both first and second player to control for advantages resulting
    from choosing better opening moves or having first initiative to move.
    """

    timeout_count = 0
    forfeit_count = 0
    pool = PPool(NUM_PROCS)

    for _ in range(num_matches):

        # initialize all games with a random move and response
        init_moves = []
        init_game = Board("p1", "p2")
        for _ in range(2):
            move = random.choice(init_game.get_legal_moves())
            init_moves.append(move)
            init_game.apply_move(move)

        games = sum([[(2 * i, cpu_agent.player, agent.player, init_moves),
                      (2 * i + 1, agent.player, cpu_agent.player, init_moves)]
                    for i, agent in enumerate(test_agents)], [])

        # play all games and tally the results
        for result, termination in pool.map(_run, games):
            game = games[result[0]]
            winner = game[1] if result[1] else game[2]

            win_counts[winner] += 1

            if termination == "timeout":
                timeout_count += 1
            elif winner not in test_agents and termination == "forfeit":
                forfeit_count += 1

    return timeout_count, forfeit_count


def update(total_wins, wins):
    for player in total_wins:
        total_wins[player] += wins[player]
    return total_wins


def play_matches(cpu_agents, test_agents, num_matches):
    """Play matches between the test agent and each cpu_agent individually. """
    total_wins = {agent.player: 0 for agent in test_agents}
    total_timeouts = 0.
    total_forfeits = 0.
    total_matches = 2 * num_matches * len(cpu_agents)

    print("\n{:^9}{:^13}{:^13}{:^13}{:^13}{:^13}".format(
        "Match #", "Opponent", test_agents[0].name, test_agents[1].name,
        test_agents[2].name, test_agents[3].name))
    print("{:^9}{:^13} {:^5}| {:^5} {:^5}| {:^5} {:^5}| {:^5} {:^5}| {:^5}"
          .format("", "", *(["Won", "Lost"] * 4)))

    for idx, agent in enumerate(cpu_agents):
        wins = {test_agents[0].player: 0,
                test_agents[1].player: 0,
                test_agents[2].player: 0,
                test_agents[3].player: 0,
                agent.player: 0}

        print("{!s:^9}{:^13}".format(idx + 1, agent.name), end="", flush=True)

        counts = play_round(agent, test_agents, wins, num_matches)
        total_timeouts += counts[0]
        total_forfeits += counts[1]
        total_wins = update(total_wins, wins)
        _total = 2 * num_matches
        round_totals = sum([[wins[agent.player], _total - wins[agent.player]]
                            for agent in test_agents], [])
        print(" {:^5}| {:^5} {:^5}| {:^5} {:^5}| {:^5} {:^5}| {:^5}"
              .format(*round_totals))

    print("-" * 74)
    print("{:^9}{:^13}{:^13}{:^13}{:^13}{:^13}\n".format(
        "", "Win Rate:",
        *["{:.1f}%".format(100 * total_wins[a.player] / total_matches)
          for a in test_agents]
    ))

    if total_timeouts:
        print(("\nThere were {} timeouts during the tournament -- make sure " +
               "your agent handles search timeout correctly, and consider " +
               "increasing the timeout margin for your agent.\n").format(
            total_timeouts))
    if total_forfeits:
        print(("\nYour ID search forfeited {} games while there were still " +
               "legal moves available to play.\n").format(total_forfeits))


def run_tournament(cpu_agents, test_agents, num_matches):
    """Play matches between the test agent and each cpu_agent individually. """
    total_wins = {agent.player: 0 for agent in test_agents}
    total_timeouts = 0.
    total_forfeits = 0.

    for agent in cpu_agents:
        wins = {test_agents[0].player: 0,
                test_agents[1].player: 0,
                test_agents[2].player: 0,
                test_agents[3].player: 0,
                agent.player: 0}

        counts = play_round(agent, test_agents, wins, num_matches)
        total_timeouts += counts[0]
        total_forfeits += counts[1]
        total_wins = update(total_wins, wins)

    return total_wins, total_timeouts, total_forfeits


def main():
    test_agents = [
        Agent(AlphaBetaPlayer(score_fn=improved_score), "AB_Improved"),
        Agent(AlphaBetaPlayer(score_fn=custom_score), "AB_Custom"),
        Agent(AlphaBetaPlayer(score_fn=custom_score_2), "AB_Custom_2"),
        Agent(AlphaBetaPlayer(score_fn=custom_score_3), "AB_Custom_3"),
    ]

    cpu_agents = [
        Agent(MinimaxPlayer(score_fn=open_move_score), "MM_Open"),
        Agent(MinimaxPlayer(score_fn=center_score), "MM_Center"),
        Agent(MinimaxPlayer(score_fn=improved_score), "MM_Improved"),
        Agent(AlphaBetaPlayer(score_fn=open_move_score), "AB_Open"),
        Agent(AlphaBetaPlayer(score_fn=center_score), "AB_Center"),
        Agent(AlphaBetaPlayer(score_fn=improved_score), "AB_Improved")
    ]

    play_matches(cpu_agents, test_agents, NUM_MATCHES)


if __name__ == "__main__":
    main()
