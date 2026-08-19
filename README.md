# Routing Footprint and Development

Country-level analysis of Internet infrastructure against economic development, built from
address registry records, the global routing table, and active traceroute measurement.

Developed as a capstone project for the CAIDA ESCALATE program, summer 2026.

## Research question

A country's Internet footprint can be measured at four progressively narrower stages.
Address space **delegated** by a regional registry, address space **announced** in the global
routing table, destinations actually **reached** by an active probe, and the **latency** of
the path that reaches them. Each stage is a stricter claim than the one before it.

Which of these stages track economic development, and which are artifacts of how address
space is registered rather than properties of infrastructure?

## Headline findings

**Latency is the measure that works.** After absorbing which measurement vantage point
probed which target, a doubling of income per capita is associated with roughly a 9.5
percent reduction in latency to a country's network edge. R² of 0.406 across 139 countries.
The coefficient moves only from -0.137 to -0.151 as the minimum sample per country is
tightened from 30 traces to 1,000, so the result is not driven by small samples.

**Address space scales one-for-one with population.** The estimated population elasticity of
delegated IPv4 space is 1.003, and a test against exactly one returns p = 0.92. Dividing by
population is therefore justified rather than assumed. Income elasticity is 1.19, so richer
countries hold disproportionately more.

**Autonomous systems behave differently from address space.** The population elasticity for
AS counts is 0.74, well below one. Larger countries do not get proportionally more networks,
they get larger ones.

**The share of delegated space a country announces is not linear in income.** A linear model
finds nothing, R² of 0.012 with the income coefficient at p = 0.89. Adding a quadratic term
raises R² to 0.082 with the squared term at p = 0.0002, and the curve peaks near 9,000
dollars per capita. Two different groups sit at the low end for opposite reasons. Poor
countries such as Cameroon at 0.40 cannot use the space they hold. Large financial centres
such as Singapore at 0.60 and Luxembourg at 0.73 hold space that was never operationally
deployed. A linear model averages the two into nothing.

**Registration data is a good proxy for routed footprint.** Residuals computed from delegated
space and from announced space correlate at 0.987, and the income elasticity barely moves
between them. Delegated files, which are public and archived back to 2003, are therefore
defensible for longitudinal work where historical geolocation data is not available.

## Two case studies

**Iran.** 0.074 of traces reach the destination country, across 1,967 traces. Iran announces
93 percent of its delegated space, so this is not scarcity. Paths terminate before entering
the country, which is consistent with centralised filtering infrastructure.

**Seychelles.** 73 IPv4 addresses per resident, roughly fifteen times the United States rate,
and the largest positive residual in the delegated-space regression at 5.37 in logs. Offshore
registration rather than infrastructure. The same pattern appears independently in the
announcement and reachability measures.

## Methodological findings

These were as informative as the results themselves.

**Requiring a probe to reach its destination fails.** About 90 percent of traces end in
GAPLIMIT rather than a reply from the target, because routers commonly decline to answer
while still forwarding traffic. Defining reachability by the last router that responded, and
mapping that hop into the delegation blocks, raises the usable share of traces from roughly
10 percent to 70 percent.

**Strict proximity measures firewall policy, not infrastructure.** Requiring the last
responding hop to fall in the destination's own /24 puts France at 0.041 and Ireland at
0.052, both well connected countries. That measure captures whether operators answer probes.

**The measurement apparatus has to be controlled for.** Raw latency partly reflects which
monitor happened to draw a given target. Raw and monitor-adjusted latency correlate at 0.873,
so the vantage point matters without driving the result. Monitor fixed effects are used
throughout.

**Geolocation error is correlated with the regressor.** Commercial IP geolocation fails at
substantially higher rates in the Global South than the Global North. Because income is the
main regressor, that is non-classical measurement error rather than noise. RIR delegated
files are used as the primary country attribution to avoid it, and the same attribution basis
is held constant across every stage so that differences between stages are real.

## Figures

| | |
|---|---|
| `figures/01_latency_income.png` | Adjusted latency against income, the main result |
| `figures/02_announcement_ratio.png` | Announcement share against income, linear and quadratic fits, offshore centres marked |
| `figures/03_stages_by_income.png` | Each stage of the hierarchy by World Bank income group |

## Data sources

**RIR delegated extended statistics files** from AFRINIC, APNIC, ARIN, LACNIC, and RIPE NCC.
Primary country attribution. 260,709 IPv4 ranges, 70,342 IPv6 ranges, 102,313 AS number
ranges. Historical snapshots retrieved yearly from 2003 to 2026.

**CAIDA RouteViews prefix-to-AS**, used for announced address space. Announced prefixes
overlap, since operators announce covering prefixes alongside more specific ones, so the
union of announced ranges is used rather than the sum. The naive sum overstates by a factor
of 1.368.

**CAIDA Ark IPv4 Routed /24 Topology**, list 7 allpref24, team 1. One cycle, 19 monitors
across six continents, 603,000 traces.

**World Bank World Development Indicators**, for GDP, GDP per capita, population, and
individuals using the Internet. Most recent non-null year per country per indicator across
2010 to 2025.

**Zoromé (2007), IMF Working Paper 07/87**, for the offshore financial centre classification.
An externally published list is used so that jurisdictions are not selected because they fit
the pattern in this data.

## Known limitations

Taiwan holds 35.7 million delegated addresses and appears in the routing and traceroute data,
but is absent from World Bank indicators and therefore drops from every regression.

Announced prefixes are attributed by their starting address, so a prefix spanning two
delegated blocks is assigned entirely to the first. This is rare.

Two of the 19 monitors sit in the United Kingdom and several in the United States. Monitor
fixed effects absorb each monitor's average but not the physical proximity of British and
American destinations to those monitors, so individual rich-country positions should be read
with more caution than the overall slope.

The offshore financial centre category is heterogeneous. Most listed jurisdictions announce
95 to 99 percent of their space. The low-announcement behaviour belongs to the large financial
hubs specifically, which is why the offshore coefficient is only marginally significant at
p = 0.06.

## Pipeline

Scripts run in order from the repository root.

```
src/parse_delegated.py        registry files to per-country resource counts
src/fetch_worldbank.py        World Bank indicators
src/build_panel.py            join registry and economic data
src/run_regression.py         log-log regressions and residuals
src/attribute_announced.py    announced space, union of overlapping prefixes
src/analyze_gap.py            delegated versus announced
src/test_ofc.py               non-linear income and offshore effects
src/fetch_historical.py       yearly delegated snapshots, 2003 to 2026
src/build_trace_table.py      Ark traces to delegation blocks
src/merge_reachability.py     three reachability definitions
src/monitor_fixed_effects.py  monitor-adjusted country effects
src/make_figures.py           figures
```

Outputs are written to `results/` and are committed. Raw and intermediate data are not.

Ark data requires an approved CAIDA topology data request. `sc_warts2json` from the scamper
distribution is required to read warts files.

## Data citation

The CAIDA UCSD IPv4 Routed /24 Topology Dataset - 2026-08-11,
https://www.caida.org/catalog/datasets/ipv4_routed_24_topology_dataset/

The CAIDA UCSD Macroscopic Internet Topology Data Kit (ITDK) - 2026,
https://www.caida.org/catalog/datasets/internet-topology-data-kit/

The CAIDA UCSD RouteViews Prefix to AS mappings Dataset - 2026-08-11,
https://catalog.caida.org/dataset/routeviews_prefix2as