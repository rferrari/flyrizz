# Fly Speed Dating (backend)

Real Drosophila-connectome courtship decision loop. Player tunes a "pheromone blend" (real
`ORN_DA1`/`ORN_DA2` cVA-glomerulus intensities); the same real connectome (`male-cns:v1.0`)
evaluates it and, via real `pC1`/`aSP` courtship decision-hub neurons, drives real `DNp13`
(courtship-acceptance) and `DNa01` (steering) descending-neuron outputs.

Per the design: there is only ever **one** real connectome instance. "Male" and "Female" are
purely cosmetic labels on the same brain -- the male panel is the sensory-input tuning UI, the
female panel is how that same brain's decision output is displayed. No second brain is faked or
fetched.

See `/home/adam/.claude/plans/greedy-bouncing-flamingo.md` for the full design/verification
rationale (every neuron type here -- `pC1`, `aSP`, `DNp13`, `DNa01`, `ORN_DA1`, `ORN_DA2` -- was
confirmed to actually exist and connect via live NeuPrint queries, not assumed).
