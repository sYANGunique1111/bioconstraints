---
trigger: always_on
---

Import packages rules:
- For embedder classes, define in pose_embedder.py
- For classic modules (MHSA, MHCA, MLP, etc,), define in basic_modules.py
- Avoid redundant class definitions. Reuse existing classes by importing them from their defined source locations.
