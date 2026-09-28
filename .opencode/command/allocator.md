---
description: Run the autonomous value-investing allocator - daily idea or monthly rebalance proposal.
agent: value-allocator
---

Run the allocator in the requested mode: $ARGUMENTS

- If the argument is `daily` (or empty), follow the **Daily procedure** in your
  instructions: research exactly one new investment idea and persist it to
  `.state/`.
- If the argument is `monthly`, follow the **Monthly procedure**: run the refresh
  gate, compare holdings and ideas via the `portfolio-comparison` subagent,
  derive the account-2 and account-3 allocations from `allocator/policy.json`,
  write the proposal files under `.state/proposals/<YYYY-MM>/`, dry-run both
  rebalances, notify, and stop. Never execute.

Finish by summarising what was produced and where it is, plus the next action
required from the user.
