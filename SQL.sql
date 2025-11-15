WITH ordered AS (
  SELECT
    EMPLOYEE_ID,
    REPORT_DATE,
    HR_EMPLOYEE_STATUS_DESC,
    -- Mark start of a new segment whenever status changes (or first row)
    CASE
      WHEN LAG(HR_EMPLOYEE_STATUS_DESC) OVER (
             PARTITION BY EMPLOYEE_ID
             ORDER BY REPORT_DATE
           ) = HR_EMPLOYEE_STATUS_DESC
      THEN 0
      ELSE 1
    END AS is_new_segment
  FROM `your_dataset.your_table`
),

segmented AS (
  SELECT
    EMPLOYEE_ID,
    REPORT_DATE,
    HR_EMPLOYEE_STATUS_DESC,
    -- Running sum of "new segment" flags → unique segment id per employee
    SUM(is_new_segment) OVER (
      PARTITION BY EMPLOYEE_ID
      ORDER BY REPORT_DATE
    ) AS segment_id
  FROM ordered
)

SELECT
  EMPLOYEE_ID,
  MIN(REPORT_DATE) AS HR_EMPLOYEE_STATUS_START_DATE,
  HR_EMPLOYEE_STATUS_DESC,
  MAX(REPORT_DATE) AS HR_EMPLOYEE_STATUS_END_DATE
FROM segmented
GROUP BY
  EMPLOYEE_ID,
  HR_EMPLOYEE_STATUS_DESC,
  segment_id
ORDER BY
  EMPLOYEE_ID,
  HR_EMPLOYEE_STATUS_START_DATE;
