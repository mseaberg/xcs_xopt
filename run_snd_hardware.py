"""Standalone Xopt optimization of the XCS SnD alignment on the real hardware.

This runs the *same* optimization code that hutch-python drives interactively,
but as a plain script so you can tweak Xopt settings and rerun without paying the
multi-minute hutch-python restart. It imports ``User`` from ``xcs_xopt`` (the
hardware device container), connects to just the SnD EPICS devices, runs the
chosen optimizer, and saves/plots the result.

The optimization path does NOT need hutch-python's ``RE`` / databroker -- only
``dscan_and_fit`` does, and that is not used here.

Settings come from the CONSTANTS block below; any CLI flag overrides its
constant. Booleans support a --no-<name> form (e.g. --no-prior, --no-move).

Run (from the repo directory):
    python run_snd_hardware.py                 # uses the constants below
    python run_snd_hardware.py --prior         # attach the physics prior
    python run_snd_hardware.py --method turbo --num-iter 200
    python run_snd_hardware.py --sim           # FastMotor dry run, no crystal motion

WARNING: without --sim this moves the real SnD crystal motors.
"""

import argparse

import numpy as np

from xcs_xopt import User


# ---- defaults (overridable on the command line) ----------------------------
METHOD = "bo"            # "bo" (UCB) or "turbo" (EI + trust region)
USE_PRIOR = False        # attach the differentiable-sim physics prior (real motors only)
MOTION_RANGE = 50e-6     # angular search range per knob (rad)
N_INIT = 64              # random initial samples
NUM_ITER = 150           # optimizer steps after init
SCALE = None             # objective intensity weight; None -> 1.0 (turbo) / 1e-4 (bo)
SEED = 42                # BO initial-sample seed (turbo ignores it)
SIM = False              # True -> simulated FastMotors (does not move real crystals)
MOVE_TO_OPTIMUM = True   # move to the best sample when finished
PLOT = True              # show the convergence plot at the end


def main(args):
    scale = args.scale
    if scale is None:
        scale = 1.0 if args.method == "turbo" else 1e-4

    print(
        "[run_snd_hardware] "
        f"{'SIM (FastMotor)' if args.sim else 'REAL SnD motors'} | "
        f"method={args.method} | prior={args.prior} | "
        f"motion_range={args.motion_range} rad | n_init={args.n_init} | "
        f"num_iter={args.num_iter} | scale={scale}"
    )

    user = User()
    user.set_motors(motion_range=args.motion_range, sim=args.sim)
    user.set_target()
    print(
        f"centroid target: x={user.optimizer.x_target}, "
        f"y={user.optimizer.y_target}"
    )

    if args.prior:
        # must run after set_target (prior captures the target) and before
        # initialize_* (prior is wired in when the generator's model is built)
        user.enable_prior(intensity_scale=scale)
        print("physics prior attached")

    if args.method == "turbo":
        user.initialize_turbo(n_init=args.n_init, scale=scale)
        user.run_turbo(num_iter=args.num_iter)
    else:
        user.initialize_BO(n_init=args.n_init, scale=scale)
        user.run_BO(num_iter=args.num_iter, seed=args.seed)

    # get_optimum_details saves the run CSV/NPZ to the optimizer's savepath
    user.get_optimum_details(plot=args.plot, move_to_optimum=args.move_to_optimum)
    return user


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["bo", "turbo"], default=METHOD)
    p.add_argument("--prior", action=argparse.BooleanOptionalAction,
                   default=USE_PRIOR,
                   help="attach the physics prior (real motors only)")
    p.add_argument("--sim", action=argparse.BooleanOptionalAction, default=SIM,
                   help="drive simulated FastMotors instead of the real crystals")
    p.add_argument("--motion-range", type=float, default=MOTION_RANGE,
                   help="angular search range per knob (rad)")
    p.add_argument("--n-init", type=int, default=N_INIT)
    p.add_argument("--num-iter", type=int, default=NUM_ITER)
    p.add_argument("--scale", type=float, default=SCALE,
                   help="objective intensity weight (default: per-method)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--move-to-optimum", action=argparse.BooleanOptionalAction,
                   default=MOVE_TO_OPTIMUM)
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=PLOT)
    args = p.parse_args()

    if args.prior and args.sim:
        p.error("--prior requires the real motors; it is not available with --sim")
    return args


if __name__ == "__main__":
    main(parse_args())
