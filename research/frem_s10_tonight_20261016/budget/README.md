# Isolated account budget guard

This stdlib utility is an engineering appendix. It makes no cloud calls, installs
no packages, launches no jobs and does not activate GPUs. `allocation_enabled`
defaults to `False`; unverified account readiness independently blocks allocation.
Wharton/S10 is CPU only. The observed account UI and phone-verification state are
not converted into a live GPU authorization or a fabricated charge here.

Policy uses a caller-supplied initial remaining quota Q0, with no default. Unknown
Q0 blocks. Preferred cap is `max(0, min(6, Q0 - 24))`; explicitly approved expanded
cap is `max(0, min(10, Q0 - 20))`. Ten percent of that cap is withheld, so only 90%
is spendable. The absolute stop deadline is 2026-10-16 16:30:00 UTC. Every run has
an all-in timeout covering setup, compute, export and shutdown. Wall seconds,
device-hours, estimated quota-hours and observed remaining quota-hours are
separate fields. Explicit conversion evidence and all four phase bounds are
required. The account floor is always 20 remaining quota-hours.

Create one ledger per account/quota period and share its exact path among every
participant. `reserve` atomically records the upper cost. It is not a launch
authorization. `authorize_launch` consumes that reservation once, rechecks fresh
quota, account-wide empty active-session and coordination observations, and returns
a stop time. Call immediately before platform launch. `assert_within_timeout`
must be integrated into the runner's watchdog; this module does not itself enforce
external scheduling or kill a process. A process kill/exit does not establish that
a paid platform session stopped. Record an outcome, then independently obtain a
fresh account-wide platform-stop observation after the reserved interval.

Failures, interruptions and unused reservations retain their entire estimated
upper cost for this campaign. No estimated amount is labeled an observed GPU
charge. No automatic refund or refill is supported, even after platform-stop
confirmation, because provider billing can lag. The floor check additionally
subtracts all prior estimates from the latest observed remaining quota; this can
double count already billed usage and intentionally blocks some affordable work.
Observed quota consumption also counts against the frozen campaign cap. A quota
reset/increase or observed runtime overrun blocks further work pending review.

Ledger updates hold a POSIX `flock` across read/validate/append, use hash-linked
events, and fsync each append. Existing ledgers are never recreated or truncated.
A partial append, broken chain or clock regression fails closed. Concurrent
reservations admit at most one unresolved run. The ledger is local coordination,
not a provider-wide lock: all account users must cooperate on this ledger and
provide genuine account-wide observations. File hashes detect accidental edits,
not a malicious writer able to rewrite the whole ledger. Evidence IDs and UTC
times are caller attestations, not authenticated provider receipts. A trustworthy
UTC clock, truthful conversion bounds and platform-aware stop integration remain
caller responsibilities. The test receipt covers simulated stdlib fixtures only.
