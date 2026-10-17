# Northstar Support Console

Agent-driven SaaS mock for Stage-B **video → DAG** data. **Three procedures** → three DAG families.

| Procedure | Paths | Show different DAG |
|-----------|-------|--------------------|
| **warranty** | active / expired | Warranty decision → repair vs paid |
| **parts** | in_stock / backorder | Stock decision → ship vs backorder |
| **escalate** | p1 / standard | Severity → on-call vs L2 queue |

Protocol: **[AGENT.md](AGENT.md)**

```bash
python3 -m http.server 8765
# http://127.0.0.1:8765/?procedure=warranty&path=active&hud=0
# http://127.0.0.1:8765/?procedure=parts&path=in_stock&hud=0
# http://127.0.0.1:8765/?procedure=escalate&path=p1&hud=0
```

```js
Northstar.procedures()
Northstar.act("set_procedure", { procedure: "parts", path: "backorder" })
Northstar.playbook()
```
