PRAGMA enable_progress_bar=false;

WITH RECURSIVE gd(i,a1,a2,a3,a4,a5,a6,b) AS (
  SELECT 1,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE
  UNION ALL
  SELECT
    s.i + 1,
    s.a1 - 0.1*AVG(mg.g.d1),
    s.a2 - 0.1*AVG(mg.g.d2),
    s.a3 - 0.1*AVG(mg.g.d3),
    s.a4 - 0.1*AVG(mg.g.d4),
    s.a5 - 0.1*AVG(mg.g.d5),
    s.a6 - 0.1*AVG(mg.g.d6),
    s.b  - 0.1*AVG(mg.g.d7)
  FROM (
    SELECT
      gd.i,
      gd.a1 AS a1, gd.a2 AS a2, gd.a3 AS a3, gd.a4 AS a4, gd.a5 AS a5, gd.a6 AS a6, gd.b AS b,
      gd.a1 AS w1, gd.a2 AS w2, gd.a3 AS w3, gd.a4 AS w4, gd.a5 AS w5, gd.a6 AS w6, gd.b AS wb,
      d.x1  AS f1, d.x2  AS f2, d.x3  AS f3, d.x4  AS f4, d.x5  AS f5, d.x6  AS f6, d.y6 AS yy
    FROM gd
    CROSS JOIN data d
    WHERE gd.i < 20
  ) s
  CROSS JOIN LATERAL (
    SELECT mygrad(
      s.w1,s.w2,s.w3,s.w4,s.w5,s.w6,s.wb,
      s.f1,s.f2,s.f3,s.f4,s.f5,s.f6,s.yy,
      (p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14) ->
        (p1*p8 + p2*p9 + p3*p10 + p4*p11 + p5*p12 + p6*p13 + p7 - p14)
        *
        (p1*p8 + p2*p9 + p3*p10 + p4*p11 + p5*p12 + p6*p13 + p7 - p14)
    ) AS g
  ) mg
  GROUP BY s.i, s.a1, s.a2, s.a3, s.a4, s.a5, s.a6, s.b
)
SELECT * FROM gd WHERE i=20;
