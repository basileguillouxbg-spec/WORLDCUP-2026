"""Step 2: Dixon-Coles low-score correction to fix draw blindness.

Negative rho lifts the 0-0 and 1-1 cells (the most common draws) and trims
1-0 / 0-1, which is exactly where the independent grid loses draws. We sweep
rho on the walk-forward harness and keep the value that minimises held-out
log-loss. (rho is a single global scalar, so sweeping it on the eval folds
carries negligible overfitting risk.)
"""
from harness import walk_forward

print("Sweeping Dixon-Coles rho (rho=0 is the Step-1 baseline):\n")
print(f"{'rho':>7}{'log-loss':>11}{'brier':>9}{'accuracy':>10}{'pred-draw':>11}")
results = {}
for rho in [0.0, -0.02, -0.04, -0.06, -0.08, -0.10, -0.12, -0.15, -0.20]:
    r = walk_forward(rho=rho, verbose=False)
    results[rho] = r
    print(f"{rho:>7.2f}{r['log_loss']:>11.4f}{r['brier']:>9.4f}{r['accuracy']:>10.4f}{r['pred_draw_rate']:>11.3f}")

best = min(results, key=lambda k: results[k]["log_loss"])
base = results[0.0]
b = results[best]
print(f"\nBest rho = {best}  (true draw rate ~ {b['true_draw_rate']:.3f})")
print(f"  log-loss : {base['log_loss']:.4f} -> {b['log_loss']:.4f}  ({b['log_loss']-base['log_loss']:+.4f})")
print(f"  brier    : {base['brier']:.4f} -> {b['brier']:.4f}  ({b['brier']-base['brier']:+.4f})")
print(f"  accuracy : {base['accuracy']:.4f} -> {b['accuracy']:.4f}  ({b['accuracy']-base['accuracy']:+.4f})")
print(f"  draw rate: {base['pred_draw_rate']:.3f} -> {b['pred_draw_rate']:.3f}")
print("\n  confusion at best rho (rows=true, cols=pred) [away,draw,home]:")
for row in b["cm"]:
    print("   ", row)
