# Laura Gao client mandate

Verified case facts are recorded in `docs/CASE_CONTRACT.md` and sourced to the
hashed four-page case PDF. The executable contract is in `rejepa.mandate`.

The default schedule is $300,000 at the beginning of 2027, $150,000 at the
beginning of 2028, no withdrawals before 2033, and ten $50,000 beginning-of-year
operating payments from 2033 through 2042. No WInS gain or loss is inserted into
the long-term asset path.

`LiabilitySchedule` computes reserve present values. `LiquidityReserve` protects
flexibility. `FundingFrontier` reports shortfall, reliability and facility-bound
scenarios. `FacilityContributionPolicy` requires configurable confidence and
reliability targets. `dynamic_sleeve_weights` divides capital among:

- **FUND:** capital allocated toward operating obligations;
- **GUARD:** cash and flexibility reserve, increased by forecast uncertainty;
- **GROW:** surplus risk capacity, scaled by declared preference.

These are planning calculations, not promised returns. Facility cost, tax,
Taiwan legal requirements and official risk preferences remain outside the
case evidence and cannot be invented.
