
DROP TABLE IF EXISTS t1e6;

CREATE TABLE t1e6 AS
SELECT (i+1)::DOUBLE AS x1,               -- > 0
       (i % 100 + 1)::DOUBLE AS x2,       -- 1..100
       (i % 7   + 1)::DOUBLE AS x3        -- 1..7
FROM range(1000000) t(i);


COPY (
  SELECT x1*x1 + x1*x2 + pow(x2,3) - x3/x1 AS y
  FROM t1e6
) TO 'C:\Users\lenov\Documents\baseline_out.csv' (HEADER, DELIMITER ',');


PRAGMA enable_profiling = json;
PRAGMA profiling_output = 'C:\Users\lenov\Documents\baseline_profile.json';

EXPLAIN ANALYZE
SELECT x1*x1 + x1*x2 + pow(x2,3) - x3/x1 AS y
FROM t1e6;

PRAGMA disable_profiling;
