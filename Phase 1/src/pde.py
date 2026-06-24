import QuantLib as ql
from option import Option


def pde_crank_nicolson(option: Option, style: str = 'american', n_steps: int = 200, n_space: int = 200) -> float:
    """Price an option using the Crank-Nicolson finite difference method via QuantLib.

    Parameters:
        option: Option dataclass with S, K, T, r, sigma, kind.
        style:  'american' or 'european'.
        n_steps: Number of time steps for the finite difference grid.
        n_space: Number of space steps for the finite difference grid.
    Returns:
        The option price.
    """
    today = ql.Date.todaysDate()
    maturity = today + ql.Period(int(option.T * 365), ql.Days)
    ql.Settings.instance().evaluationDate = today

    dc = ql.Actual365Fixed()
    process = ql.BlackScholesProcess(
        ql.QuoteHandle(ql.SimpleQuote(option.S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(option.r)), dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(option.sigma)), dc)),
    )

    payoff_type = ql.Option.Call if option.kind == 'call' else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(payoff_type, option.K)

    if style == 'american':
        exercise = ql.AmericanExercise(today, maturity)
    else:
        exercise = ql.EuropeanExercise(maturity)

    vanilla = ql.VanillaOption(payoff, exercise)
    vanilla.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, n_steps, n_space))
    return vanilla.NPV()


if __name__ == "__main__":
    opt = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='put')
    price = pde_crank_nicolson(opt, style='american')
    print(f"Prix de l'option via PDE Crank-Nicolson : {price:.4f}")