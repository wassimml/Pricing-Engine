import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from option import Option
from pricer import BSModel


def parse_args():
    parser = argparse.ArgumentParser(
        prog="pricer_terminal",
        description="Pricing Engine — valorisation d'options européennes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        méthodes disponibles:
        bs              Formule analytique Black-Scholes-Merton
        mc-naive        Monte Carlo naïf
        mc-antithetic   Monte Carlo avec variables antithétiques
        mc-control               Monte Carlo avec variable de contrôle
        mc-control-antithetic    Monte Carlo antithétique + variable de contrôle

        exemples:
        python pricer_terminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind call
        python pricer_terminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method mc-antithetic --n-paths 200000
                """,
    )

    parser.add_argument("--S",       type=float, required=True, metavar="SPOT",   help="Prix spot du sous-jacent")
    parser.add_argument("--K",       type=float, required=True, metavar="STRIKE", help="Prix d'exercice")
    parser.add_argument("--T",       type=float, required=True, metavar="YEARS",  help="Maturité en années (ex: 0.5, 1)")
    parser.add_argument("--r",       type=float, required=True, metavar="RATE",   help="Taux sans risque continu (ex: 0.05)")
    parser.add_argument("--sigma",   type=float, required=True, metavar="VOL",    help="Volatilité annualisée (ex: 0.2)")
    parser.add_argument("--kind",    type=str,   required=True, choices=["call", "put"])
    parser.add_argument("--method",  type=str,   default="bs",
                        choices=["bs", "mc-naive", "mc-antithetic", "mc-control", "mc-control-antithetic"],
                        help="Méthode de pricing (défaut: bs)")
    parser.add_argument("--n-paths", type=int,   default=100_000, metavar="N",   help="Nombre de trajectoires MC (défaut: 100000)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    opt  = Option(S=args.S, K=args.K, T=args.T, r=args.r, sigma=args.sigma, kind=args.kind)

    if args.method == "bs":
        price = BSModel().price(opt)
        print(f"BS Price ({args.kind}): {price:.6f}")

    elif args.method == "mc-naive":
        from monteCarlo import mc_naive
        price = mc_naive(opt, n_paths=args.n_paths)
        print(f"MC Naive ({args.kind}, n={args.n_paths:,}): {price:.6f}")

    elif args.method == "mc-antithetic":
        from monteCarlo import mc_antithetic
        price = mc_antithetic(opt, n_paths=args.n_paths)
        print(f"MC Antithetic ({args.kind}, n={args.n_paths:,}): {price:.6f}")

    elif args.method == "mc-control":
        from monteCarlo import mc_control
        price = mc_control(opt, n_paths=args.n_paths)
        print(f"MC Control Variate ({args.kind}, n={args.n_paths:,}): {price:.6f}")

    elif args.method == "mc-control-antithetic":
        from monteCarlo import mc_control_antithetic
        price = mc_control_antithetic(opt, n_paths=args.n_paths)
        print(f"MC Control + Antithetic ({args.kind}, n={args.n_paths:,}): {price:.6f}")
