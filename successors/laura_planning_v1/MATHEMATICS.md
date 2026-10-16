# Laura planning arithmetic: conditional derivations

These statements describe `core.py` and its explicit assumptions. They prove
accounting identities and conditional bounds. They do **not** prove market
performance, investment eligibility, executable prices, a future protected
surplus, or a calibrated 99% funding probability. Source/date fields are
attestations; arithmetic cannot authenticate them. Tests are separate execution
evidence and are not reported as run by this derivation review.

All amounts below are nominal USD at the indicated date. Inputs are interpreted
as exact rational numbers using `Fraction`; the rounded dollar display never
feeds back into calculations. `money_record` explicitly uses half-even decimal
rounding. Exact components conserve wealth even when individually rounded cents
do not add to the displayed rounded total.

## 1. Beginning-year contributions

Let `C_t` be a contribution available at the beginning of calendar year `t`,
`r_t >= -1` that year's simple net return, and `W_t` wealth immediately before
the contribution. With no interim withdrawals,

\[
W_{t+1}=(W_t+C_t)(1+r_t).
\]

Repeated substitution, starting from zero wealth, gives

\[
W_T=\sum_{t<T} C_t\prod_{u=t}^{T-1}(1+r_u).
\]

Thus the case contributions produce wealth immediately before the first 2033
payment:

\[
W_{2033}=300000\prod_{u=2027}^{2032}(1+r_u)
 +150000\prod_{u=2028}^{2032}(1+r_u).
\]

Under a constant assumed annual net return `g`, this is
`300000(1+g)^6 + 150000(1+g)^5`. At zero return it is USD 450,000.
The 2028 contribution earns five years, not six, and is not 2027 buying power.
A total loss (`r_t=-1`) erases invested wealth at that point; later contributions
can still enter. Returns below -1 are rejected for this unlevered cash-flow
model. Reordering annual returns can change wealth because contributions enter
at different times.

`required_constant_growth` brackets the rate required to reach a stated target;
it does not predict that rate. The expression is continuous and strictly
increasing on `g>-1`, with derivative
`1800000(1+g)^5 + 750000(1+g)^4 > 0`. Its value at `g=-1` is zero.
Bisection therefore preserves a root bracket whenever the initial endpoint
values contain the target. An out-of-bracket target or unmet numerical tolerance
is reported instead of being converted into a successful forecast.

## 2. The immediate operating payment belongs in the reserve

At the beginning of 2033, the ten nominal USD 50,000 payments occur at offsets
`0,1,...,9`, ending at the beginning of 2042. For an assumed flat annual discount
rate `d>-1`,

\[
R_{2033}(d)=50000\sum_{j=0}^{9}(1+d)^{-j}.
\]

The first term is exactly USD 50,000. At `d=0`, `R=500000`; with one payment,
`R=50000` for every admissible rate. There is no return after the last payment in
this formula. Subtracting the first payment again before reserving this amount
would count it twice.

More generally, if `D_j` is the valuation-date discount factor for payment
`L_j`, then `R=sum(L_j D_j)`, requiring every liability date, positive factors,
and `D_0=1` for an immediate liability. These factors can be declared scenarios.
They establish an executable reserve only if actual accessible cash flows,
settlement, taxes/costs, and payment availability support the match. A quoted
yield or a rolling bill-fund yield alone is not such evidence. The date-only
beginning-year convention does not resolve holidays or banking deadlines.

## 3. Backward funding of a specified payment path

For nonnegative payments `L_0,...,L_(n-1)` and strictly positive between-payment
growth factors `1+q_j`, define

\[
B_{n-1}=L_{n-1},\qquad
B_j=L_j+\frac{B_{j+1}}{1+q_j},\quad j=n-2,\ldots,0.
\]

Starting with `B_j`, paying `L_j` leaves `B_(j+1)/(1+q_j)`; applying the stated
growth factor gives exactly `B_(j+1)` for the next payment. Induction funds every
payment, leaves nonnegative balances between payments, and ends at zero after
the last payment. Exactly `n-1` growth assumptions are required. This proves
funding **if those growth factors occur**, not that risky returns can be locked.
Zero or negative growth factors are rejected because this backward division
cannot finance a later positive payment through total loss.

## 4. Costs, reserve, flexibility, and facility conserve wealth

Let `V>=0` be available wealth, `K>=0` required incremental essential
implementation costs, `R>=0` the operating reserve requirement, `a in [0,1]`
the retained residual fraction, and `b>=0` a requested minimum cash floor.
Costs already deducted from net returns or included in the reserve price must
not appear in `K` a second time.

The implementation allocates in this order:

\[
\begin{aligned}
K_p&=\min(V,K),& N&=V-K_p,\\
R_p&=\min(N,R),& S_+&=N-R_p=\max(V-K-R,0),\\
B^*&=\max(b,aS_+),& B&=\min(S_+,B^*),\\
F&=S_+-B.
\end{aligned}
\]

Every allocated component is nonnegative. Direct substitution proves

\[
\boxed{V=K_p+R_p+B+F.}
\]

Required-but-unfunded amounts remain separate: `D_K=K-K_p`, `D_R=R-R_p`, and
`D_B=B*-B`. The signed economic surplus is

\[
S=V-K-R,\qquad D_K+D_R=\max(-S,0).
\]

This signed quantity must deduct **required** costs, even when only part can be
paid. For `V=1000, K=1500, R=50000`, it is `-50500`, with USD 500 cost shortfall
and USD 50,000 reserve shortfall. The code retains zero facility capacity and
does not borrow to conceal the deficit.

The flexibility formula takes the greater of the percentage and minimum,
rather than adding two buffers. Since `B>=aS_+`, `F<=(1-a)S_+`. If `b=0`,
`B=aS_+` and `F=(1-a)S_+` exactly. With the proposed `a=0.2`, USD 600,000 wealth,
USD 10,000 incremental costs and USD 500,000 reserve give USD 90,000 surplus,
USD 18,000 retained flexibility and USD 72,000 facility capacity.

The separate strategic-sleeve rule uses
`E=min(p S_+, c N, max(S_+-H,0))`, where `H>=0` is any secured surplus floor encumbered for a communicated facility minimum (zero before such a commitment), for `p,c in [0,1]`, and liquidity `Q=S_+-E`.
Thus `V=K_p+R_p+E+Q`, `E<=S_+`, and `E<=cN`: the equity target cannot consume
the allocated operating reserve, and its cap is measured on post-cost wealth.
These are target dollars conditional on a defensible reserve price, not orders
or evidence that the reserve can already be purchased.

## 5. What a 2031 update conditions on

For a genuinely observed beginning-2031 balance `B_2031>=0`, with scenario `s`
containing exactly calendar-year returns `r_(2031,s)` and `r_(2032,s)`,

\[
W_{2033,s}=B_{2031}(1+r_{2031,s})(1+r_{2032,s}).
\]

Each resulting wealth is paired with the same named scenario's reserve and
incremental costs before applying the allocation identity above. This preserves
joint wealth/reserve/cost uncertainty. It is not valid to independently choose
the most favorable wealth and cheapest reserve from incompatible scenarios.

`conditional_2031_range` requires the observation date to be 2031-01-01, its
availability to be no later than the decision, a decision in 2031, and assumptions
declared by that decision. It represents the **beginning-year balance** even if
the decision is later in 2031; it is not an arbitrary-date current-NAV adapter.
`illustrative_2031_range` instead uses a hypothetical balance and requires a
decision before 2031-01-01. A 2026 calculation therefore cannot acquire an
observed-2031 label.

Minima and maxima across named scenarios are just extrema of those assumptions.
They do not imply statistical coverage, calibrated probabilities, or that every
economically possible outcome was included. Both a missing confidence estimate
and unfunded scenarios are preserved explicitly.

## 6. Conditional facility floor and communicated cap

`bounded_facility_commitment` implements the percentage-only policy (`b=0`).
It does not implement an additional fixed cash minimum. Let `S` be realized
terminal surplus **after the operating reserve and all relevant costs**, `H>=0`
an independently secured surplus floor, `a in [0,1]` the retained fraction, and
`U>=0` the communicated facility cap. Define

\[
L=(1-a)H,\qquad U\ge L,\qquad
G(S)=\min\{(1-a)\max(S,0),U\}.
\]

If the protected-floor condition `S>=H` holds, then `S>=0` and
`(1-a)S >= (1-a)H=L`. Both arguments of the minimum are at least `L`, proving

\[
\boxed{L\le G(S)\le U.}
\]

Since `(1-a)max(S,0)<=max(S,0)`, remaining flexibility
`B_c=max(S,0)-G(S)` is nonnegative, satisfies
`B_c>=a max(S,0)`, and conserves positive surplus:
`G(S)+B_c=max(S,0)`. Upside above the cap remains available rather than increasing
the communicated gift. When `a=1`, both the gift and promised floor are zero.
When `H=0`, the theorem gives no positive minimum gift.

For a full-equity-loss stress, suppose the protected net surplus is `H` and all
remaining exposure is an unlevered sleeve of current value `E>=0`, with terminal
net value `E_T>=0`. If the protection survives and no omitted claim can invade
it, terminal surplus is `S=H+E_T>=H`. Even at total equity loss `E_T=0`,
`G=L`. For example, `H=40000`, `a=0.2`, and `U=80000` give a conditional minimum
gift of USD 32,000 and USD 8,000 retained flexibility at total equity loss.
At `S=200000`, the gift is capped at USD 80,000 and USD 120,000 remains.

The hypothesis is essential. A declared `H` does not secure it. The condition
can fail through inaccessible or late proceeds, issuer nonpayment, leveraged
losses or recourse, currency mismatch, sale before protected cash flows arrive,
unfunded implementation/tax costs, incorrect reserve pricing, or unmodeled
withdrawals. The proof makes no claim that these events have zero probability.
No particular investment has been established here to satisfy the hypothesis.

If `S<H`, the function marks `SECURED_SURPLUS_FLOOR_BREACHED`; for `a<1` the gift
also falls below `L` when `H>0`. For `S<0`, the signed deficit is retained, gift
and flexibility are zero, and the shortfall does not disappear through clipping.
For instance `S=20000,H=40000,a=0.2` gives USD 16,000, explicitly below the
USD 32,000 conditional floor. With `S=-5000`, the USD 5,000 deficit remains.

The function reports the caller-attested, unauthenticated basis of `H`, sets
funding probability to null, and denies an unconditional guarantee. A valid
conditional inequality is therefore neither a proof of superior returns nor
a calibrated 99% promise for Laura.

## 7. Rebalancing counterexample and protected surplus

With a USD 500,000 operating allocation and discretionary USD 60,000 equities plus USD 40,000 cash, a 40% equity loss leaves USD 36,000 equities and USD 40,000 cash. A naive 60/40 rebalance of the USD 76,000 surplus would buy USD 9,600 equities from the protected cash. A subsequent total equity loss would leave USD 30,400 and a gift of USD 24,320, below the communicated USD 32,000 minimum.

The added equity cap `max(S_+-H,0)` prevents that purchase when H=40000: equities remain at most USD 36,000 and at least USD 40,000 stays liquid. If surplus falls below H, the code records the protected-surplus shortfall and targets zero equity; it does not pretend the earlier promise remains supported. This is a target-allocation property conditional on actual protection, not a market guarantee.
