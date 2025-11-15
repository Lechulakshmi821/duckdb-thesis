-- bench_compare.sql
-- Compare plain SQL vs mygrad_lambda on 100000 rows

-- 1) Create / recreate benchmark table with 100000 tuples
DROP TABLE IF EXISTS t;
CREATE TABLE t AS
SELECT
    i::DOUBLE AS x,
    (i + 1)::DOUBLE AS y
FROM range(100000) AS t(i);

-- 2) Check row count (sanity)
SELECT COUNT(*) AS nrows FROM t;

-- 3) Baseline: plain SQL expression
SELECT
    SUM(x*x + x*y + y*y) AS baseline_sum
FROM t;

-- 4) mygrad_lambda: same expression via lambda + AD
--    IMPORTANT: here lambda is passed as a STRING,
--    because mygrad_lambda is defined as (struct, varchar).

SELECT
    SUM(
        struct_extract(
            mygrad_lambda(
                struct_pack(x := x, y := y),
                '(x,y) -> x*x + x*y + y*y'
            ),
            'result'
        )
    ) AS mygrad_sum
FROM t;
