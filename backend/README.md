# FlyRizz (backend)

Real Drosophila-connectome courtship decision game. Pick a role (Male Suitor or Female Chooser),
mint a fly (randomly rolled cosmetic traits + a real "pheromone blend" on `ORN_DA1`/`ORN_DA2`,
hidden from the player), then swipe on candidates. Each date's real outcome is judged by the same
real connectome (`male-cns:v1.0`): the suitor's broadcast blend, scaled by the other fly's own
hidden receptiveness, drives real `pC1`/`aSP` courtship decision-hub neurons, which in turn drive
real `DNp13` (courtship-acceptance) and `DNa01` (steering) descending-neuron outputs. A MATCH needs
both the player's swipe and the real brain's judgment to agree.

Per the design: there is only ever **one** real connectome instance. "Male" and "Female" are
purely cosmetic role labels on the same brain -- no second brain is faked or fetched. Cosmetic
traits (Wing Shimmer, Leg Length, etc.) and per-fly color tint are flavor only, not derived from
any real neuron; only DA1/DA2, pC1/aSP, DNp13, and DNa01 are real.

See `/home/adam/.claude/plans/greedy-bouncing-flamingo.md` for the full design/verification
rationale (every neuron type here -- `pC1`, `aSP`, `DNp13`, `DNa01`, `ORN_DA1`, `ORN_DA2` -- was
confirmed to actually exist and connect via live NeuPrint queries, not assumed).
