#!/bin/bash
set -e

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
    SELECT gd.i, gd.a1, gd.a2, gd.a3, gd.a4, gd.a5, gd.a6, gd.b,
      mygrad_fwd(
        gd.a1,gd.a2,gd.a3,gd.a4,gd.a5,gd.a6,gd.b,
        data.x1,data.x2,data.x3,data.x4,data.x5,data.x6,data.y6,
        1.0::DOUBLE,
        (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15) ->
          -(p14*ln(p15/(p15+exp(-(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7))))
           +(p15-p14)*ln(p15-p15/(p15+exp(-(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7)))))
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
    SELECT gd.i, gd.a1, gd.a2, gd.a3, gd.a4, gd.a5, gd.a6, gd.b,
      mygrad_rev(
        gd.a1,gd.a2,gd.a3,gd.a4,gd.a5,gd.a6,gd.b,
        data.x1,data.x2,data.x3,data.x4,data.x5,data.x6,data.y6,
        1.0::DOUBLE,
        (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15) ->
          -(p14*ln(p15/(p15+exp(-(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7))))
           +(p15-p14)*ln(p15-p15/(p15+exp(-(p1*p8+p2*p9+p3*p10+p4*p11+p5*p12+p6*p13+p7)))))
      ) AS g
    FROM gd, data WHERE gd.i < 20
  )
  GROUP BY i,a1,a2,a3,a4,a5,a6,b
)
SELECT 'rev_logistic' AS method, * FROM gd WHERE i=20;
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
    SELECT gd.i, gd.a1, gd.a2, gd.a3, gd.a4, gd.a5, gd.a6, gd.b,
      mygrad_fwd(
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
    SELECT gd.i, gd.a1, gd.a2, gd.a3, gd.a4, gd.a5, gd.a6, gd.b,
      mygrad_rev(
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

echo "Fixed all 4 AD benchmark files"
