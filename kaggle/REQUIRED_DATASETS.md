# Exact external data

Attach [Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs), owner/slug `borismarjanovic/price-volume-data-for-all-us-stocks-etfs`.
The provider card was rechecked 25 October 2026 and lists CC0. This is the
provider's license assertion, not independent rights certification or proof of
historically available adjustments.

`input_manifest.json` records every exact seven-ETF member SHA-256. The adapter
requires one canonical `Data/ETFs/<ticker>.us.txt` per asset under `/kaggle/input`.
Basename matches under `Stocks` do not substitute. No separate source or weight
dataset is needed: the notebook embeds the hash-bound archive. Dataset size and
current mount version have not been reverified in an authenticated target.

Use the retained SPY/IWM/DIA/TLT/IEF/GLD/SLV source bytes only. QQQ exclusion was
prospectively recorded after a calendar-only failure; this is a seven-ETF
successor, not a repaired completion of the original eight-ETF study. Present-day
adjusted-close ratios are historical proxies, with no verified publication
vintage, survivorship completeness, terminal outcomes or executable prices.
