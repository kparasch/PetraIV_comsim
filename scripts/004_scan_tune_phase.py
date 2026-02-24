import argparse
from pySC import SimulatedCommissioning
import numpy as np
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=1)
    argparser.add_argument('--threads', type=int, default=1)
    args = argparser.parse_args()

    seed = args.seed

    SC = SimulatedCommissioning.from_json(f'data/Seeds/pySC_petra4_03_seed{seed}.json')

    threads = args.threads if args.threads > 1 else None

    SC.injection.n_particles = 50
    SC.lattice.omp_num_threads = threads

    def scan_q(deltas, mode='+', n_turns=10, compare_last_turn=False):
        best_eff = 0
        best_delta = 0
        threshold = 0.4
        if mode == '+':
            sx = 1
            sy = 1
        elif mode == '-':
            sx = 1
            sy = -1
        elif mode == 'x':
            sx = 1
            sy = 0
        elif mode == 'y':
            sx = 0
            sy = 1
        for delta in deltas:
            SC.tuning.tune.trim(sx*delta, sy*delta)
            tr = SC.tuning.injection_efficiency(n_turns=n_turns)
            if not np.any(tr < threshold):
                max_turns = n_turns
            else:
                max_turns = np.argmax(tr < threshold)
            if compare_last_turn:
                if tr[-1] > best_eff:
                    best_eff = tr[-1]
                    best_delta = delta
                elif tr[-1] == best_eff:
                    if abs(delta) < abs(best_delta):
                        best_eff = tr[-1]
                        best_delta = delta
                print(f"Tune ({mode}) {delta:.2f}, max eff. {best_eff*100}% / {n_turns} turns")
            else:
                if max_turns > best_eff:
                    best_eff = max_turns
                    best_delta = delta
                elif max_turns == best_eff:
                    if abs(delta) < abs(best_delta):
                        best_eff = max_turns
                        best_delta = delta
                print(f"Tune ({mode}) {delta:.2f}, max turns {max_turns}")
            SC.tuning.tune.trim(-sx*delta, -sy*delta)
        print(f"Best (Q{mode}) delta: {best_delta}")
        return best_delta

    n_turns=100
    amp_delta = 0.3
    deltas = np.linspace(-amp_delta, amp_delta, 7)
    delta = scan_q(deltas, mode='+', n_turns=n_turns)
    SC.tuning.tune.trim(delta, delta)
    delta = scan_q(deltas, mode='-', n_turns=n_turns)
    SC.tuning.tune.trim(delta, -delta)
    delta = scan_q(deltas, mode='x', n_turns=n_turns)
    SC.tuning.tune.trim(delta, 0)
    delta = scan_q(deltas, mode='y', n_turns=n_turns)
    SC.tuning.tune.trim(0, delta)
    n_turns=200
    amp_delta /= 2
    deltas = np.linspace(-amp_delta, amp_delta, 7)
    delta = scan_q(deltas, mode='+', n_turns=n_turns)
    SC.tuning.tune.trim(delta, delta)
    delta = scan_q(deltas, mode='-', n_turns=n_turns)
    SC.tuning.tune.trim(delta, -delta)

    SC.tuning.rf.optimize_phase(low=-10, high=10, npoints=21, n_turns=100)

    n_turns=400

    amp_delta = 0.1
    delta = scan_q(deltas, mode='+', n_turns=n_turns, compare_last_turn=True)
    SC.tuning.tune.trim(delta, delta)
    delta = scan_q(deltas, mode='-', n_turns=n_turns, compare_last_turn=True)
    SC.tuning.tune.trim(delta, -delta)
    n_turns=800
    amp_delta /= 2
    delta = scan_q(deltas, mode='+', n_turns=n_turns, compare_last_turn=True)
    SC.tuning.tune.trim(delta, delta)
    delta = scan_q(deltas, mode='-', n_turns=n_turns, compare_last_turn=True)
    SC.tuning.tune.trim(delta, -delta)



    SC.injection.n_particles = 512
    injef = SC.tuning.injection_efficiency(n_turns=1000)
    for ii, tr in enumerate(injef):
        if (ii + 1) % 100 == 0:
            print(f"Turn {ii+1}, {tr*100:.0f}% transmission")

    SC.to_json(f'data/Seeds/pySC_petra4_04_seed{seed}.json')
    np.savez(f"data/Seeds/injection_efficiency_before_orbit{seed}", injef)
