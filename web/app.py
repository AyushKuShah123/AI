
import requests
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template, jsonify
import base64
from io import BytesIO
from datetime import datetime

app = Flask(__name__)

# ThingSpeak API endpoint
api_url = "https://api.thingspeak.com/channels/2473122/fields/{field}.json?results=10"

# CSV files to store sensor data
csv_files = ['CSV_file/field1.csv', 'CSV_file/field2.csv', 'CSV_file/field3.csv', 'CSV_file/field4.csv']

# Field names
field_names = ['Temperature', 'Humidity', 'Soil Moisture', 'Water Flow Status']
line_colors = ['blue', 'green', 'orange', 'red']  # Different colors for each line

# Retrieve data from ThingSpeak API and store in CSV files
def get_data(field):
    try:
        response = requests.get(api_url.format(field=field))
        response.raise_for_status()
        data = response.json()
        sensor_data = [(entry['created_at'], entry['field' + str(field)]) for entry in data['feeds']]
        
        # Append new data to CSV
        with open(csv_files[field - 1], 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(sensor_data)
        
        return sensor_data
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return []

# Initialize sensor data
for i in range(1, 5):
    get_data(i)

# Create figures and axes for graphs
figs, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Display current date and time
date_time_text = plt.gcf().text(0.5, 0.95, "", fontsize=18, ha='center')

def display_current_time():
    now = datetime.now().strftime("%Y-%m-%d   %H:%M:%S")
    date_time_text.set_text(now)

# Update graphs with new data
def update(i):
    display_current_time()
    all_table_data = []
    
    for j, ax in enumerate(axes):
        ax.clear()
        ax.set_xlabel('Time')
        ax.set_ylabel(field_names[j])
        ax.set_title(field_names[j])
        
        data = []
        with open(csv_files[j], 'r') as csvfile:
            reader = csv.reader(csvfile)
            all_data = list(reader)[-6:]  # Get the last 6 rows
            for row in all_data:
                if row[0] and row[1]:
                    data.append((mdates.datestr2num(row[0]), float(row[1])))
                    all_table_data.append([row[0], row[1] if j == 0 else '', row[1] if j == 1 else '', row[1] if j == 2 else '', row[1] if j == 3 else ''])
        
        if data:
            ax.plot([x[0] for x in data], [x[1] for x in data], marker='o', linestyle='-', color=line_colors[j])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_xlim([data[0][0], data[-1][0]])
            ax.legend()

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    return all_table_data

# Flask route to return graphs as HTML and table data
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to update CSV files with new data
@app.route('/update', methods=['GET'])
def update_data():
    for i in range(1, 5):
        get_data(i)
    return jsonify({'message': 'Data updated'})

# Flask route to return graphs and table data
@app.route('/graph', methods=['GET'])
def get_graph():
    table_data = update(0)  # Update data for graph
    buf = BytesIO()
    figs.savefig(buf, format='png')
    buf.seek(0)
    graph = base64.b64encode(buf.read()).decode("ascii")
    
    return jsonify({'graph': graph, 'data': table_data})

if __name__ == '__main__':
    app.run(debug=True)

