#!/bin/bash
set -e

cat > bench_baseline_logistic.sql.in << 'EOF'
PRAGMA threads=1;
DROP TABLE IF EXISTS data;
CREATE TABLE data AS
WITH base AS (
  SELECT
    random()::DOUBLE AS x1, random()::DOUBLE AS x2, random()::DOUBLE AS x3,
    random()::DOUBLE AS x4, random()::DOUBLE AS x5, random()::DOUBLE AS x6
  FROM range(__N__)
)
SELECT x1, x2, x3, x4, x5, x6,
  CASE WHEN random() < 1.0/(1.0+exp(-(0.5+0.4*x1+0.25*x2+0.1*x3+0.2*x4+0.6*x5+0.7*x6)))
       THEN 1.0 ELSE 0.0 END::DOUBLE AS y6
FROM base;

WITH RECURSIVE gd(i,a1,a2,a3,a4,a5,a6,b) AS (
  SELECT 1, 1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE
  UNION ALL
  SELECT i+1,
    a1 - 0.1*AVG((p-y6)*x1), a2 - 0.1*AVG((p-y6)*x2),
    a3 - 0.1*AVG((p-y6)*x3), a4 - 0.1*AVG((p-y6)*x4),
    a5 - 0.1*AVG((p-y6)*x5), a6 - 0.1*AVG((p-y6)*x6),
    b  - 0.1*AVG(p-y6)
  FROM (
    SELECT gd.i, gd.a1, gd.a2, gd.a3, gd.a4, gd.a5, gd.a6, gd.b,
      1.0/(1.0+exp(-(gd.a1*data.x1+gd.a2*data.x2+gd.a3*data.x3+gd.a4*data.x4+gd.a5*data.x5+gd.a6*data.x6+gd.b))) AS p,
      data.x1, data.x2, data.x3, data.x4, data.x5, data.x6, data.y6
    FROM gd, data WHERE gd.i < 20
  )
  GROUP BY i,a1,a2,a3,a4,a5,a6,b
)
SELECT 'baseline_logistic' AS method, * FROM gd WHERE i=20;
EOF

cat > bench_fwd_logistic.sql.in << 'EOF'
PRAGMA threads=1;
DROP TABLE IF EXISTS data;
CREATE TABLE data AS
WITH base AS (
  SELECT
    random()::DOUBLE AS x1, random()::DOUBLE AS x2, random()::DOUBLE AS x3,
    random()::DOUBLE AS x4, random()::DOUBLE AS x5, random()::DOUBLE AS x6
  FROM range(__N__)
)
SELECT x1, x2, x3, x4, x5, x6,
  CASE WHEN random() < 1.0/(1.0+exp(-(0.5+0.4*x1+0.25*x2+0.1*x3+0.2*x4+0.6*x5+0.7*x6)))
       THEN 1.0 ELSE 0.0 END::DOUBLE AS y6
FROM base;

WITH RECURSIVE gd(i,a1,a2,a3,a4,a5,a6,b) AS (
  SELECT 1, 1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE
  UNION ALL
  SELECT i+1,
    a1-0.1*AVG(g.d1), a2-0.1*AVG(g.d2), a3-0.1*AVG(g.d3), a4-0.1*AVG(g.d4),
    a5-0.1*AVG(g.d5), a6-0.1*AVG(g.d6), b -0.1*AVG(g.d7)
  FROM (
    SELECT mygrad_fwd(
      gd.a1,gd.a2,gd.a3,gd.a4,gd.a5,gd.a6,gd.b,
      data.x1,data.x2,data.x3,data.x4,data.x5,data.x6,data.y6,
      (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14) ->
        -1.0*(p14*ln(1.0/(1.0+exp(-1.0*(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7))))
             +(1.0-p14)*ln(1.0-1.0/(1.0+exp(-1.0*(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7)))))
    ) AS g
    FROM gd, data WHERE gd.i < 20
  )
  GROUP BY i,a1,a2,a3,a4,a5,a6,b
)
SELECT 'fwd_logistic' AS method, * FROM gd WHERE i=20;
EOF

cat > bench_rev_logistic.sql.in << 'EOF'
PRAGMA threads=1;
DROP TABLE IF EXISTS data;
CREATE TABLE data AS
WITH base AS (
  SELECT
    random()::DOUBLE AS x1, random()::DOUBLE AS x2, random()::DOUBLE AS x3,
    random()::DOUBLE AS x4, random()::DOUBLE AS x5, random()::DOUBLE AS x6
  FROM range(__N__)
)
SELECT x1, x2, x3, x4, x5, x6,
  CASE WHEN random() < 1.0/(1.0+exp(-(0.5+0.4*x1+0.25*x2+0.1*x3+0.2*x4+0.6*x5+0.7*x6)))
       THEN 1.0 ELSE 0.0 END::DOUBLE AS y6
FROM base;

WITH RECURSIVE gd(i,a1,a2,a3,a4,a5,a6,b) AS (
  SELECT 1, 1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE
  UNION ALL
  SELECT i+1,
    a1-0.1*AVG(g.d1), a2-0.1*AVG(g.d2), a3-0.1*AVG(g.d3), a4-0.1*AVG(g.d4),
    a5-0.1*AVG(g.d5), a6-0.1*AVG(g.d6), b -0.1*AVG(g.d7)
  FROM (
    SELECT mygrad_rev(
      gd.a1,gd.a2,gd.a3,gd.a4,gd.a5,gd.a6,gd.b,
      data.x1,data.x2,data.x3,data.x4,data.x5,data.x6,data.y6,
      (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14) ->
        -1.0*(p14*ln(1.0/(1.0+exp(-1.0*(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7))))
             +(1.0-p14)*ln(1.0-1.0/(1.0+exp(-1.0*(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7)))))
    ) AS g
    FROM gd, data WHERE gd.i < 20
  )
  GROUP BY i,a1,a2,a3,a4,a5,a6,b
)
SELECT 'rev_logistic' AS method, * FROM gd WHERE i=20;
EOF

cat > bench_baseline_poisson.sql.in << 'EOF'
PRAGMA threads=1;
DROP TABLE IF EXISTS data;
CREATE TABLE data AS
WITH base AS (
  SELECT
    random()::DOUBLE AS x1, random()::DOUBLE AS x2, random()::DOUBLE AS x3,
    random()::DOUBLE AS x4, random()::DOUBLE AS x5, random()::DOUBLE AS x6
  FROM range(__N__)
)
SELECT x1, x2, x3, x4, x5, x6,
  round(exp(0.5*(0.5+0.4*x1+0.25*x2+0.1*x3+0.2*x4+0.6*x5+0.7*x6)))::DOUBLE AS y6
FROM base;

WITH RECURSIVE gd(i,a1,a2,a3,a4,a5,a6,b) AS (
  SELECT 1, 1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE
  UNION ALL
  SELECT i+1,
    a1 - 0.01*AVG((p-y6)*x1), a2 - 0.01*AVG((p-y6)*x2),
    a3 - 0.01*AVG((p-y6)*x3), a4 - 0.01*AVG((p-y6)*x4),
    a5 - 0.01*AVG((p-y6)*x5), a6 - 0.01*AVG((p-y6)*x6),
    b  - 0.01*AVG(p-y6)
  FROM (
    SELECT gd.i, gd.a1, gd.a2, gd.a3, gd.a4, gd.a5, gd.a6, gd.b,
      exp(gd.a1*data.x1+gd.a2*data.x2+gd.a3*data.x3+gd.a4*data.x4+gd.a5*data.x5+gd.a6*data.x6+gd.b) AS p,
      data.x1, data.x2, data.x3, data.x4, data.x5, data.x6, data.y6
    FROM gd, data WHERE gd.i < 20
  )
  GROUP BY i,a1,a2,a3,a4,a5,a6,b
)
SELECT 'baseline_poisson' AS method, * FROM gd WHERE i=20;
EOF

cat > bench_fwd_poisson.sql.in << 'EOF'
PRAGMA threads=1;
DROP TABLE IF EXISTS data;
CREATE TABLE data AS
WITH base AS (
  SELECT
    random()::DOUBLE AS x1, random()::DOUBLE AS x2, random()::DOUBLE AS x3,
    random()::DOUBLE AS x4, random()::DOUBLE AS x5, random()::DOUBLE AS x6
  FROM range(__N__)
)
SELECT x1, x2, x3, x4, x5, x6,
  round(exp(0.5*(0.5+0.4*x1+0.25*x2+0.1*x3+0.2*x4+0.6*x5+0.7*x6)))::DOUBLE AS y6
FROM base;

WITH RECURSIVE gd(i,a1,a2,a3,a4,a5,a6,b) AS (
  SELECT 1, 1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE
  UNION ALL
  SELECT i+1,
    a1-0.01*AVG(g.d1), a2-0.01*AVG(g.d2), a3-0.01*AVG(g.d3), a4-0.01*AVG(g.d4),
    a5-0.01*AVG(g.d5), a6-0.01*AVG(g.d6), b -0.01*AVG(g.d7)
  FROM (
    SELECT mygrad_fwd(
      gd.a1,gd.a2,gd.a3,gd.a4,gd.a5,gd.a6,gd.b,
      data.x1,data.x2,data.x3,data.x4,data.x5,data.x6,data.y6,
      (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14) ->
        exp(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7) - p14*(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7)
    ) AS g
    FROM gd, data WHERE gd.i < 20
  )
  GROUP BY i,a1,a2,a3,a4,a5,a6,b
)
SELECT 'fwd_poisson' AS method, * FROM gd WHERE i=20;
EOF

cat > bench_rev_poisson.sql.in << 'EOF'
PRAGMA threads=1;
DROP TABLE IF EXISTS data;
CREATE TABLE data AS
WITH base AS (
  SELECT
    random()::DOUBLE AS x1, random()::DOUBLE AS x2, random()::DOUBLE AS x3,
    random()::DOUBLE AS x4, random()::DOUBLE AS x5, random()::DOUBLE AS x6
  FROM range(__N__)
)
SELECT x1, x2, x3, x4, x5, x6,
  round(exp(0.5*(0.5+0.4*x1+0.25*x2+0.1*x3+0.2*x4+0.6*x5+0.7*x6)))::DOUBLE AS y6
FROM base;

WITH RECURSIVE gd(i,a1,a2,a3,a4,a5,a6,b) AS (
  SELECT 1, 1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE
  UNION ALL
  SELECT i+1,
    a1-0.01*AVG(g.d1), a2-0.01*AVG(g.d2), a3-0.01*AVG(g.d3), a4-0.01*AVG(g.d4),
    a5-0.01*AVG(g.d5), a6-0.01*AVG(g.d6), b -0.01*AVG(g.d7)
  FROM (
    SELECT mygrad_rev(
      gd.a1,gd.a2,gd.a3,gd.a4,gd.a5,gd.a6,gd.b,
      data.x1,data.x2,data.x3,data.x4,data.x5,data.x6,data.y6,
      (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14) ->
        exp(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7) - p14*(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7)
    ) AS g
    FROM gd, data WHERE gd.i < 20
  )
  GROUP BY i,a1,a2,a3,a4,a5,a6,b
)
SELECT 'rev_poisson' AS method, * FROM gd WHERE i=20;
EOF

cat > run_loss_function_bench.sh << 'EOF'
#!/bin/bash
set -e
DUCKDB_BIN="./build/release/duckdb"
SIZES=(1000 5000 10000 20000 50000 100000 200000 500000 1000000)
RUNS=5
OUT_CSV="loss_function_scaling.csv"

if [ ! -x "$DUCKDB_BIN" ]; then
    echo "ERROR: $DUCKDB_BIN not found or not executable."
    exit 1
fi

echo "tuples,method,run,seconds" > "$OUT_CSV"

run_one() {
    local n="$1"
    local sql_file="$2"
    local method="$3"
    local tmpsql="/tmp/bench_tmp_${method}_${n}.sql"
    sed "s/__N__/${n}/g" "$sql_file" > "$tmpsql"
    for run in $(seq 1 $RUNS); do
        local start=$(date +%s.%N)
        "$DUCKDB_BIN" < "$tmpsql" > /tmp/bench_out.log 2>&1
        local end=$(date +%s.%N)
        local elapsed=$(awk "BEGIN {printf \"%.4f\", $end - $start}")
        echo "${n},${method},${run},${elapsed}" >> "$OUT_CSV"
        echo "  [$method] N=$n run=$run: ${elapsed}s"
    done
}

for n in "${SIZES[@]}"; do
    echo "=== N=$n ==="
    run_one "$n" bench_baseline_logistic.sql.in baseline_logistic
    run_one "$n" bench_fwd_logistic.sql.in      fwd_logistic
    run_one "$n" bench_rev_logistic.sql.in      rev_logistic
    run_one "$n" bench_baseline_poisson.sql.in  baseline_poisson
    run_one "$n" bench_fwd_poisson.sql.in       fwd_poisson
    run_one "$n" bench_rev_poisson.sql.in       rev_poisson
done

echo "DONE - raw data in $OUT_CSV"
EOF
chmod +x run_loss_function_bench.sh

cat > aggregate_loss.py << 'EOF'
import csv
from collections import defaultdict
import statistics

rows = defaultdict(list)
with open('loss_function_scaling.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (int(row['tuples']), row['method'])
        rows[key].append(float(row['seconds']))

methods = ['baseline_logistic', 'fwd_logistic', 'rev_logistic',
           'baseline_poisson', 'fwd_poisson', 'rev_poisson']
sizes = sorted(set(k[0] for k in rows.keys()))

print(f"{'tuples':>9} | " + " | ".join(f"{m:>18}" for m in methods))
print("-" * 140)
with open('loss_function_summary.csv', 'w') as out:
    out.write("tuples," + ",".join(f"{m}_median" for m in methods) + "\n")
    for n in sizes:
        medians = []
        for m in methods:
            vals = rows.get((n, m), [])
            med = statistics.median(vals) if vals else float('nan')
            medians.append(med)
        print(f"{n:>9} | " + " | ".join(f"{v:>18.4f}" for v in medians))
        out.write(f"{n}," + ",".join(f"{v:.4f}" for v in medians) + "\n")

print("\nSummary written to loss_function_summary.csv")
EOF

echo "All 8 files created successfully:"
ls -la bench_*.sql.in run_loss_function_bench.sh aggregate_loss.py
