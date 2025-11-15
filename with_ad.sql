
COPY (
  SELECT mygrad_ad_eval(
           x1, x2, x3,
           '(x1,x2,x3)-> x1*x1 + x1*x2 + pow(x2,3) - x3/x1'
         ) AS y
  FROM t1e6
) TO 'C:\Users\lenov\Documents\with_ad_out.csv' (HEADER, DELIMITER ',');


PRAGMA enable_profiling = json;
PRAGMA profiling_output = 'C:\Users\lenov\Documents\with_ad_profile.json';

EXPLAIN ANALYZE
SELECT mygrad_ad_eval(
         x1, x2, x3,
         '(x1,x2,x3)-> x1*x1 + x1*x2 + pow(x2,3) - x3/x1'
       ) AS y
FROM t1e6;

PRAGMA disable_profiling;
