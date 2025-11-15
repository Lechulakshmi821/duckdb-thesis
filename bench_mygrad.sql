.timer on
PRAGMA threads=4;

-- 1) fresh data
DROP TABLE IF EXISTS t;
CREATE TABLE t AS
SELECT
  i::DOUBLE AS x,
  (i % 100)::DOUBLE AS y
FROM range(100000) t(i);

-- 2) warmup
SELECT COUNT(*) FROM t;

-- 3) run mygrad per row, write outputs to a table
DROP TABLE IF EXISTS bench_out;
CREATE TABLE bench_out AS
SELECT
  x, y,
  mygrad_lambda(STRUCT_PACK(x:=x,y:=y), '(x,y)-> x*x + x*y + y*y') AS val
FROM t;

-- 4) produce a small timing table (row count + dummy agg)
DROP TABLE IF EXISTS bench_time;
CREATE TABLE bench_time AS
SELECT COUNT(*) AS nrows, SUM(x+y) AS checksum FROM t;

-- 5) sanity peek
SELECT * FROM bench_out LIMIT 3;
SELECT * FROM bench_time;
