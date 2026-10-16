# S10 prospective event-clock clarification CL-15

15 October 2026. Revision-3 addendum to the preserved design revision 2. This resolves A4's required forced sale before the next policy decision. No S10 market values, forecasts or ledgers have been evaluated. The primary algorithm, margins, data windows, controls and 192-ledger count remain unchanged.

At a scheduled A4 rebalance open q, first mark entering shares and credit their entering-share dividends exactly once. Then execute the already-precommitted mandatory full sale at that identical opening quote in an ordered `forced_sale` subphase at q. This sale has no forecast-dependent choice. Record its separate event ID, shares, sale fee and post-sale cash. Only afterward does A4's policy choose its new allocation using the resulting all-cash NAV at q+1 microsecond; reentry fills at q+2 microseconds using the same ideal quote and incurs its own actual entry fee.

The quote/dividend and forced-sale events share q but have distinct ordered phase identities. Duplicate event IDs or an attempted forced sale from any other phase fail. Other policies do not gain this subphase. No dividend is credited on newly purchased shares. Final liquidation is already final and cannot trigger reentry. This is ideal research accounting, not executable order timing.

Implementation acceptance: immutable pure transition from OPEN to OPEN with zero shares, separately booked sell fee and unchanged quote/date; next ordinary rebalance uses the lower post-sale NAV. Independent cases must cover a dividend at the endpoint, adverse prices, zero positions, exact sell+buy fees, duplicate subphase rejection and no second dividend credit.

This is a prospective numerical/event-semantic clarification recorded before implementation and any outcomes, not an engineering-equivalence claim. The original design files and source declaration are retained; a future scientific freeze must include this addendum.
