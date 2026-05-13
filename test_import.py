import sys
import traceback
try:
    import src.config as cfg
    print("config OK, BOUNDED_REP_USE_LP =", cfg.BOUNDED_REP_USE_LP)
except Exception as e:
    print("config FAIL:", e)
    traceback.print_exc()

try:
    import src.fairness as f
    print("fairness OK")
except Exception as e:
    print("fairness FAIL:", e)
    traceback.print_exc()
