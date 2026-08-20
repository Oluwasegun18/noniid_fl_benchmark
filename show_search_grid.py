from pathlib import Path
import yaml

from flbench.optimization.search_space import grid_candidates


cfg = yaml.safe_load(
    Path("configs/controlled_search_grid.yaml").read_text(
        encoding="utf-8"
    )
)

print("Exhaustive grid sizes per scenario:")
total = 0
for algorithm in cfg["algorithms"]:
    count = len(grid_candidates(cfg, algorithm))
    total += count
    print(f"{algorithm:10s}: {count}")

print(
    "Total search configurations per "
    f"dataset/severity scenario: {total}"
)
print(
    "Total across 5 datasets x 3 severities: "
    f"{total * 15}"
)
