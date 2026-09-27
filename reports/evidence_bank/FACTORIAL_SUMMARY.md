# Forecast × controller factorial (synthetic lockbox seed)

**Status: INVALID AS EVIDENCE (protocol/DECISIONS.md D-054).** Forecasts are never passed to the controllers (both forecast rows are identical); `mpc_cvar` scenarios at step t are centred on the return it then earns (look-ahead); single 40-step path on the spent lockbox seed. Retained for provenance only.

Costs are **one-way** turnover × research bps. Base = **10 bps**.

| forecast | controller | bps | terminal_wealth | shortfall_mean |
|---|---|---:|---:|---:|
| persistence | myopic_equal | 10 | 0.4222 | 0.000000 |
| persistence | mpc_cvar | 10 | 0.4328 | 0.000000 |
| ridge | myopic_equal | 10 | 0.4222 | 0.000000 |
| ridge | mpc_cvar | 10 | 0.4328 | 0.000000 |

## Cost sensitivity (ridge × mpc_cvar)

| bps | terminal_wealth | shortfall_mean |
|---:|---:|---:|
| 0 | 0.4345 | 0.000000 |
| 5 | 0.4336 | 0.000000 |
| 10 | 0.4328 | 0.000000 |
| 25 | 0.4302 | 0.000000 |
| 50 | 0.4261 | 0.000000 |
