from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt


from streamlit_echarts import st_echarts

def impl(data_dict,val):
    if type(data_dict[val][0]) in ['FLOAT64','INT64']:
        options = {
            "yAxis": {"type": val},
            "series": [
                {"data": data_dict[val], "type": "line"}
            ],
        }
        st_echarts(options=options)

# Replace these values with your own credentials and information
project_id = 'prj-meg-dev-de-course-01'
dataset_id = 'streaming'
table_id = 'test_table01'
# Initialize a BigQuery client
client = bigquery.Client()

# Construct the SQL query to fetch data from the table
sql_query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"

# Execute the query and fetch the results
query_job = client.query(sql_query)
results = query_job.result()

# Get the schema of the table
table_ref = client.dataset(dataset_id).table(table_id)
table = client.get_table(table_ref)
schema = table.schema

# Convert the results and schema to a dictionary
data_dict = {'schema': [], 'values': []}

# Extract schema information
for field in schema:
    data_dict['schema'].append({
        'name': field.name,
        'type': field.field_type,
        'mode': field.mode
    })

data_dict = {}

# Extract values
for row in results:
    row_dict = {}
    for i, field in enumerate(schema):
        row_dict[field.name] = row[i]
    
    # Organize data by schema field
    for field in schema:
        field_name = field.name
        if field_name not in data_dict:
            data_dict[field_name] = []

        data_dict[field_name].append(row_dict[field_name])

# Convert the dictionary to a Pandas DataFrame
df = pd.DataFrame(data_dict)
print(df)
df.to_csv('bigquery_data.csv', index=False)

for val in data_dict:
    impl(data_dict,val)


