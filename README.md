# Routing Footprint and Development

Country-level analysis of Internet routing footprint against economic development.

## Research question

Does a country's routing footprint, measured as announced prefixes, advertised address space, and active ASes, track its level of economic development once population and Internet penetration are controlled for? The primary output is the residual, meaning countries whose measured infrastructure is substantially larger or smaller than their income level alone would predict.

## Data sources

RouteViews and RIPE RIS BGP RIB snapshots, parsed with BGPKit.

RIR delegated extended statistics files from AFRINIC, APNIC, ARIN, LACNIC, and RIPE NCC, used as the primary country attribution source.

World Bank World Development Indicators, covering GDP, GDP per capita, population, and individuals using the Internet.

IPinfo, used as a secondary geolocation source for validating country attribution.

CAIDA Ark topology data, used for reachability measurement.

## Measurement caveat

Commercial IP geolocation fails at substantially higher rates in the Global South than the Global North, and those failure rates are correlated with income. Because income is the primary regressor here, this is non-classical measurement error rather than noise, and it is treated explicitly in the analysis rather than assumed away.

## Data citation

The CAIDA UCSD IPv4 Routed /24 Topology Dataset - <dates used>,
https://www.caida.org/catalog/datasets/ipv4_routed_24_topology_dataset/

The CAIDA UCSD Macroscopic Internet Topology Data Kit (ITDK) - <dates used>,
https://www.caida.org/catalog/datasets/internet-topology-data-kit/

## Status

In progress. Developed as a capstone project for the CAIDA ESCALATE program, summer 2026.
