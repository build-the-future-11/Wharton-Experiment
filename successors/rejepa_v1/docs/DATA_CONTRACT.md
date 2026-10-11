# International point-in-time data contract

No dataset is currently admitted.

Every observation must identify:

- stable security identifier and point-in-time approved-universe membership;
- exchange, market, sector, asset class, local currency, and base currency;
- event time, executable price time, public availability time, and outcome
  maturity time with timezone;
- raw price, distributions, splits, delisting or terminal outcome, and the
  convention used to calculate total return;
- FX rate and FX availability for non-base-currency instruments;
- volume, liquidity, cost, and tradability information needed by the strategy;
- provider, license or usage right, retrieval time, and immutable source hash.

Development preprocessing may fit only on development history. Current members
cannot be backfilled into the past. Latest revised macro values cannot replace
historical vintages. Date-only prices cannot silently become executable closes.
An adjusted-close download alone does not satisfy this contract.

The protected boundary will be defined only after lawful data are obtained and
the development portion is inspected for coverage. Protected returns must not be
printed, summarized, used for interpreter selection, or used to choose a
bootstrap block before the scientific freeze.

Official WInS eligibility is an additional execution constraint. A security can
be useful for a research benchmark yet unavailable for competition trading; the
two universes must remain separately labeled.

