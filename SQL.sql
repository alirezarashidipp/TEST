WITH
  -- Step 1: Identify the previous status for each row
  identify_changes AS (
    SELECT
      EMPLOYEE_ID,
      REPORT_DATE,
      HR_EMPLOYEE_STATUS_DESC,
      HR_EMPLOYEE_FULL_NAME,
      LAG(HR_EMPLOYEE_STATUS_DESC) OVER (
        PARTITION BY
          EMPLOYEE_ID
        ORDER BY
          REPORT_DATE
      ) AS prev_status
    FROM
      `your-project.your-dataset.your_table` -- <-- Replace with your table name
  ),

  -- Step 2: Assign a unique group ID to each continuous status segment
  assign_segment_groups AS (
    SELECT
      *,
      SUM(
        CASE
          WHEN HR_EMPLOYEE_STATUS_DESC != prev_status THEN 1
          WHEN prev_status IS NULL THEN 1
          ELSE 0
        END
      ) OVER (
        PARTITION BY
          EMPLOYEE_ID
        ORDER BY
          REPORT_DATE
      ) AS segment_group
    FROM
      identify_changes
  ),

  -- Step 3: Create the CTE with all segments (as you requested)
  all_segments AS (
    SELECT
      EMPLOYEE_ID,
      MIN(REPORT_DATE) AS HR_EMPLOYEE_STATUS_START_DATE,
      HR_EMPLOYEE_STATUS_DESC,
      MAX(REPORT_DATE) AS HR_EMPLOYEE_STATUS_END_DATE,
      HR_EMPLOYEE_FULL_NAME
    FROM
      assign_segment_groups
    GROUP BY
      EMPLOYEE_ID,
      HR_EMPLOYEE_STATUS_DESC,
      segment_group,
      HR_EMPLOYEE_FULL_NAME
  ),

  -- Step 4: Add the ROW_NUMBER() to rank the segments for each employee
  ranked_segments AS (
    SELECT
      *,
      ROW_NUMBER() OVER (
        PARTITION BY
          EMPLOYEE_ID
        ORDER BY
          -- We use the END_DATE to find the most recent segment
          HR_EMPLOYEE_STATUS_END_DATE DESC
      ) AS rn
    FROM
      all_segments
  )
-- Step 5: Filter to get ONLY the most recent segment (rn = 1)
SELECT
  EMPLOYEE_ID,
  HR_EMPLOYEE_STATUS_START_DATE,
  HR_EMPLOYEE_STATUS_DESC,
  HR_EMPLOYEE_STATUS_END_DATE,
  HR_EMPLOYEE_FULL_NAME
FROM
  ranked_segments
WHERE
  rn = 1
ORDER BY
  EMPLOYEE_ID;
