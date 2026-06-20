import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from option import Option
from pricer import BSModel


def parse_args():
    parser = argparse.ArgumentParser(
        prog="pricer_terminal",
        description="Pricing Engine — valorisation d'options européennes et américaines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        méthodes disponibles:
        bs                       Formule analytique Black-Scholes-Merton (européen uniquement)
        binomial                 Arbre binomial CRR (européen et américain)
        mc-naive                 Monte Carlo naïf
        mc-antithetic            Monte Carlo — variables antithétiques
        mc-control               Monte Carlo — variable de contrôle
        mc-control-antithetic    Monte Carlo — antithétique + variable de contrôle

        exemples:
        python pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind call
        python pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put  --method binomial --style american --steps 500
        python pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind call --method mc-antithetic --n-paths 200000
                """,
    )

    parser.add_argument("--S",       type=float, required=True, metavar="SPOT",   help="Prix spot du sous-jacent")
    parser.add_argument("--K",       type=float, required=True, metavar="STRIKE", help="Prix d'exercice")
    parser.add_argument("--T",       type=float, required=True, metavar="YEARS",  help="Maturité en années (ex: 0.5, 1)")
    parser.add_argument("--r",       type=float, required=True, metavar="RATE",   help="Taux sans risque continu (ex: 0.05)")
    parser.add_argument("--sigma",   type=float, required=True, metavar="VOL",    help="Volatilité annualisée (ex: 0.2)")
    parser.add_argument("--kind",    type=str,   required=True, choices=["call", "put"])
    parser.add_argument("--method",  type=str,   default="bs",
                        choices=["bs", "binomial", "mc-naive", "mc-antithetic", "mc-control", "mc-control-antithetic"],
                        help="Méthode de pricing (défaut: bs)")
    parser.add_argument("--style",   type=str,   default="european", choices=["european", "american"],
                        help="Style d'exercice pour binomial (défaut: european)")
    parser.add_argument("--steps",   type=int,   default=100, metavar="N",
                        help="Nombre de pas de l'arbre binomial (défaut: 100)")
    parser.add_argument("--n-paths", type=int,   default=100_000, metavar="N",
                        help="Nombre de trajectoires MC (défaut: 100000)")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Seed aléatoire pour MC (défaut: 42)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    opt  = Option(S=args.S, K=args.K, T=args.T, r=args.r, sigma=args.sigma, kind=args.kind)

    if args.method == "bs":
        price = BSModel().price(opt)
        print(f"BS Price ({args.kind}): {price:.6f}")

    elif args.method == "binomial":
        from binomial import binomial_price
        price = binomial_price(opt, n_steps=args.steps, style=args.style)
        print(f"Binomial CRR ({args.kind}, {args.style}, N={args.steps}): {price:.6f}")

    elif args.method == "mc-naive":
        from monteCarlo import mc_naive
        price = mc_naive(opt, n_paths=args.n_paths, seed=args.seed)
        print(f"MC Naive ({args.kind}, n={args.n_paths:,}): {price:.6f}")

    elif args.method == "mc-antithetic":
        from monteCarlo import mc_antithetic
        price = mc_antithetic(opt, n_paths=args.n_paths, seed=args.seed)
        print(f"MC Antithetic ({args.kind}, n={args.n_paths:,}): {price:.6f}")

    elif args.method == "mc-control":
        from monteCarlo import mc_control
        price = mc_control(opt, n_paths=args.n_paths, seed=args.seed)
        print(f"MC Control Variate ({args.kind}, n={args.n_paths:,}): {price:.6f}")

    elif args.method == "mc-control-antithetic":
        from monteCarlo import mc_control_antithetic
        price = mc_control_antithetic(opt, n_paths=args.n_paths, seed=args.seed)
        print(f"MC Control + Antithetic ({args.kind}, n={args.n_paths:,}): {price:.6f}")
