from pathlib import Path
from collections import defaultdict
import math
import numpy as np
import matplotlib.pyplot as plt

LABEL_DIR = Path("data/a2d2/training/label_2")  # label_2 폴더 안에서 실행 중이면 "./"
CLASSES = ["Car", "Truck", "Pedestrian"]

# 거리 bin (m) 원하는 대로 바꾸면 됨
BINS = [0, 10, 20, 30, 40, 50, 60, 80, 100]  # 마지막은 필요 없으면 줄여도 됨

def parse_kitti_location(tokens):
    """
    KITTI label line tokens:
    type trunc occl alpha bbox(4) dims(h,w,l) loc(x,y,z) ry
    index: 0    1    2    3     4..7        8..10      11..13  14
    => loc = tokens[11], tokens[12], tokens[13]
    """
    if len(tokens) < 15:
        return None
    try:
        x = float(tokens[11])
        y = float(tokens[12])
        z = float(tokens[13])
        return x, y, z
    except ValueError:
        return None

# bins 준비
bin_edges = np.array(BINS, dtype=float)
bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]

counts = {c: np.zeros(len(bin_edges)-1, dtype=int) for c in CLASSES}

num_files = 0
num_objs_used = 0

for fp in LABEL_DIR.glob("*.txt"):
    num_files += 1
    for line in fp.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        toks = line.split()
        cls = toks[0]
        if cls not in CLASSES:
            continue

        loc = parse_kitti_location(toks)
        if loc is None:
            continue

        x, y, z = loc
        # ✅ 거리 정의: 수평거리(전방/좌우) = sqrt(x^2 + z^2)
        dist = math.sqrt(x*x + z*z)

        # bin index 찾기
        idx = np.searchsorted(bin_edges, dist, side="right") - 1
        if 0 <= idx < len(bin_edges)-1:
            counts[cls][idx] += 1
            num_objs_used += 1

print(f"[INFO] files={num_files}, objects_used(with loc)={num_objs_used}")
print("Bins:", bin_labels)
for c in CLASSES:
    print(c, counts[c].tolist(), " total=", int(counts[c].sum()))

# =========================
# 히스토그램(막대그래프) 출력
# =========================
xpos = np.arange(len(bin_labels))
bar_w = 0.25

plt.figure(figsize=(10, 4.5))
plt.bar(xpos - bar_w, counts["Car"], width=bar_w, label="Car")
plt.bar(xpos,         counts["Truck"], width=bar_w, label="Truck")
plt.bar(xpos + bar_w, counts["Pedestrian"], width=bar_w, label="Pedestrian")

plt.xticks(xpos, bin_labels, rotation=30, ha="right")
plt.xlabel("Distance bin (m)")
plt.ylabel("Count")
plt.title("Instance counts by distance (from label_2)")
plt.legend()
plt.tight_layout()

#out_path = LABEL_DIR / "distance_histogram.png"
#plt.savefig(out_path, dpi=200)
#print(f"[INFO] Saved plot to: {out_path.resolve()}")
plt.show()
