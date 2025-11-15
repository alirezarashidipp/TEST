WITH
  -- Step 1: Identify the previous status for each row
  identify_changes AS (
    SELECT
      EMPLOYEE_ID,
      REPORT_DATE,
      HR_EMPLOYEE_STATUS_DESC,
      -- Get the status from the previous row (ordered by date) for the same employee
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
      -- Create a segment group ID by counting the number of times the status has changed.
      -- This running sum will be the same for all consecutive rows with the same status.
      SUM(
        CASE
          -- It's a new segment if the status is different from the previous one
          WHEN HR_EMPLOYEE_STATUS_DESC != prev_status THEN 1
          -- It's also a new segment if it's the very first row for an employee
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
  )
-- Step 3: Group by the employee, status, and segment ID to get the start and end dates
SELECT
  EMPLOYEE_ID,
  MIN(REPORT_DATE) AS HR_EMPLOYEE_STATUS_START_DATE,
  HR_EMPLOYEE_STATUS_DESC,
  MAX(REPORT_DATE) AS HR_EMPLOYEE_STATUS_END_DATE
FROM
  assign_segment_groups
GROUP BY
  EMPLOYEE_ID,
  HR_EMPLOYEE_STATUS_DESC,
  segment_group
ORDER BY
  EMPLOYEE_ID,
  HR_EMPLOYEE_STATUS_START_DATE;
