-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## 0. Data Setup

-- COMMAND ----------

--Setup the salary ranges

drop table if exists salary_ranges;

create table salary_ranges (
   s int,
   e int
);

insert into salary_ranges values
   (50000, 75000),
   (75000, 100000),
   (100000, 150000),
   (150000, 25000),
   (250000, 300000);

-- COMMAND ----------

-- MAGIC %scala //add input widgets
-- MAGIC dbutils.widgets.dropdown("state", "CA", Seq("CA", "OR", "WI", "NY", "IL"), "state")
-- MAGIC dbutils.widgets.remove("gender")
-- MAGIC dbutils.widgets.multiselect("gender", "female", Seq("female", "male"), "gender")

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ##1. Company Employees

-- COMMAND ----------

select * from emp

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##2. Where do employees that are paid > 280K live?

-- COMMAND ----------

select count(*), state from emp where salary > 280000 group by state

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ##3. What is the total salary for employees in a given state?
-- MAGIC (select state from dropdown on top of the page)

-- COMMAND ----------

select state, sum(salary) from emp where state = getArgument("state") group by emp.state

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## 4. Gender distribution of employees

-- COMMAND ----------

select count(*), gender from emp where group by gender

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## 5. Salary distribution by gender

-- COMMAND ----------

select s, e , gender, count(*) from emp, salary_ranges where salary > s AND salary <=e group by s, e, gender order by e

-- COMMAND ----------

