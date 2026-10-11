"""Evidence-bound human-readable Wharton strategy reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .mandate import ClientMandate, FundingFrontier


@dataclass(frozen=True)
class StrategyEvidence:
    scientific_status: str
    admitted_data: bool
    approved_universe_verified: bool
    protected_evaluation_complete: bool
    holdings: tuple[tuple[str, float], ...] = ()
    model_limitations: tuple[str, ...] = ()


def render_strategy_markdown(
    mandate: ClientMandate,
    evidence: StrategyEvidence,
    *,
    funding: FundingFrontier | None = None,
) -> str:
    allowed = (
        evidence.scientific_status == "SUPPORTED"
        and evidence.admitted_data
        and evidence.approved_universe_verified
        and evidence.protected_evaluation_complete
    )
    lines = [
        "# Laura Gao investment strategy — evidence-bound draft",
        "",
        f"Scientific status: **{evidence.scientific_status}**.",
        "",
        "## Client goals",
        "",
        f"Contributions are {', '.join(f'${amount:,.0f} at the beginning of {year}' for year, amount in mandate.contributions)}.",
        f"The plan protects {mandate.liabilities.payment_count} beginning-of-year operating payments of ${mandate.liabilities.annual_payment:,.0f} from {mandate.liabilities.first_year} through {mandate.liabilities.final_year}.",
        "Competition trading results remain separate from the long-term projection.",
        "",
        "## Investment philosophy",
        "",
        "FUND protects operating commitments, GUARD preserves liquidity and flexibility, and GROW compounds true surplus capital. Security selection must remain long-only and inside the current approved universe unless official rules say otherwise.",
        "",
        "## Funding position",
        "",
    ]
    if funding is None:
        lines.append(
            "No admitted scenario distribution exists, so funding probabilities and facility bounds are not reported."
        )
    else:
        lines.extend(
            [
                f"Operating reserve: ${funding.required_operating_reserve:,.0f}.",
                f"Liquidity reserve: ${funding.liquidity_reserve:,.0f}.",
                f"Estimated funding reliability: {funding.funding_reliability:.1%}.",
                f"Mechanical facility-contribution interval: ${funding.facility_contribution_bounds[0]:,.0f}–${funding.facility_contribution_bounds[1]:,.0f}.",
            ]
        )
    lines.extend(["", "## Proposed holdings", ""])
    if not allowed:
        lines.append(
            "**BLOCKED:** a holdings recommendation requires admitted data, verified WInS eligibility, a completed protected evaluation, and supported scientific status."
        )
    elif not evidence.holdings:
        lines.append("No holdings were supplied despite satisfied evidence gates.")
    else:
        lines.extend(f"- {asset}: {weight:.1%}" for asset, weight in evidence.holdings)
    lines.extend(["", "## Rebalancing and risk controls", ""])
    lines.append(
        "Rebalance only on the declared schedule; enforce cash, security, sector, country, currency, liquidity, turnover, CVaR and stale-data constraints. Fall back to cash when the optimization is infeasible or evidence is stale."
    )
    lines.extend(["", "## Limitations", ""])
    limitations = evidence.model_limitations or (
        "Latent simulations are conditional hypotheses, not additional historical evidence.",
        "Engineering tests do not establish forecast or portfolio performance.",
    )
    lines.extend(f"- {item}" for item in limitations)
    return "\n".join(lines) + "\n"


def write_strategy_report(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
