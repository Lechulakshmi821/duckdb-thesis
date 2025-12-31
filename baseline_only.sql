PRAGMA enable_progress_bar=false;

WITH RECURSIVE gd(i,a1,a2,a3,a4,a5,a6,b) AS (
  SELECT 1,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE,1.0::DOUBLE
  UNION ALL
  SELECT i+1,
    a1-0.1*AVG(2*x1*(a1*x1+a2*x2+a3*x3+a4*x4+a5*x5+a6*x6+b-y6)),
    a2-0.1*AVG(2*x2*(a1*x1+a2*x2+a3*x3+a4*x4+a5*x5+a6*x6+b-y6)),
    a3-0.1*AVG(2*x3*(a1*x1+a2*x2+a3*x3+a4*x4+a5*x5+a6*x6+b-y6)),
    a4-0.1*AVG(2*x4*(a1*x1+a2*x2+a3*x3+a4*x4+a5*x5+a6*x6+b-y6)),
    a5-0.1*AVG(2*x5*(a1*x1+a2*x2+a3*x3+a4*x4+a5*x5+a6*x6+b-y6)),
    a6-0.1*AVG(2*x6*(a1*x1+a2*x2+a3*x3+a4*x4+a5*x5+a6*x6+b-y6)),
    b -0.1*AVG(2*(a1*x1+a2*x2+a3*x3+a4*x4+a5*x5+a6*x6+b-y6))
  FROM gd, data
  WHERE i < 20
  GROUP BY i,a1,a2,a3,a4,a5,a6,b
)
SELECT * FROM gd WHERE i=20;
