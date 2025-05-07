import streamlit as st
import folium
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import tempfile
import math
import pandas as pd
import io
from streamlit_folium import st_folium

# --------------------- Utility Functions ---------------------

def haversine(coord1, coord2):
    R = 6371  # km
    lat1, lon1 = map(math.radians, coord1)
    lat2, lon2 = map(math.radians, coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def draw_folium_map(routes, names, coords):
    m = folium.Map(location=coords[0], zoom_start=6)
    cmap = plt.get_cmap('tab20')
    colors = [matplotlib.colors.rgb2hex(cmap(i / len(routes))[:3]) for i in range(len(routes))]

    for idx, route in enumerate(routes, 1):
        color = colors[idx % len(colors)]
        points = [coords[i] for i in route]
        folium.PolyLine(points, color=color, weight=5, tooltip=f"Vehicle {idx}").add_to(m)
        for step, i in enumerate(route):
            folium.Marker(
                location=coords[i],
                popup=f"{names[i]} (Step {step})",
                icon=folium.Icon(color='blue' if i != 0 else 'red', icon='info-sign')
            ).add_to(m)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        m.save(f.name)
        map_file = f.name

    with open(map_file, 'r', encoding='utf-8') as f:
        html = f.read()
        st.components.v1.html(html, height=600, scrolling=True)

# --------------------- Clarke & Wright VRP ---------------------

def clarke_wright_vrp(names, coords, demands, dist, capacity):
    n = len(names) - 1
    savings_list = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dist[0][i] + dist[0][j] - dist[i][j]
            savings_list.append((s, i, j))
    savings_list.sort(key=lambda x: (-x[0], x[1], x[2]))

    routes = {}
    route_list = []

    for s, i, j in savings_list:
        route_i = routes.get(i)
        route_j = routes.get(j)

        if route_i is None and route_j is None:
            load = demands[i] + demands[j]
            if load <= capacity:
                route = [i, j]
                route_list.append({'route': route, 'load': load, 'saving': s})
                routes[i] = routes[j] = route

        elif route_i and route_j is None:
            if route_i[0] == i or route_i[-1] == i:
                load = sum(demands[k] for k in route_i) + demands[j]
                if load <= capacity:
                    if route_i[0] == i:
                        route_i.insert(0, j)
                    else:
                        route_i.append(j)
                    routes[j] = route_i
                    for r in route_list:
                        if r['route'] == route_i:
                            r['load'] = load
                            r['saving'] += s

        elif route_j and route_i is None:
            if route_j[0] == j or route_j[-1] == j:
                load = sum(demands[k] for k in route_j) + demands[i]
                if load <= capacity:
                    if route_j[0] == j:
                        route_j.insert(0, i)
                    else:
                        route_j.append(i)
                    routes[i] = route_j
                    for r in route_list:
                        if r['route'] == route_j:
                            r['load'] = load
                            r['saving'] += s

        elif route_i != route_j:
            if route_i[-1] == i and route_j[0] == j:
                merged = route_i + route_j
            elif route_j[-1] == j and route_i[0] == i:
                merged = route_j + route_i
            else:
                continue
            load = sum(demands[k] for k in merged)
            if load <= capacity:
                route_list = [r for r in route_list if r['route'] not in [route_i, route_j]]
                saving_total = s
                for r in route_list:
                    if r['route'] in [route_i, route_j]:
                        saving_total += r['saving']
                route_list.append({'route': merged, 'load': load, 'saving': saving_total})
                for k in merged:
                    routes[k] = merged

    for i in range(1, n + 1):
        if i not in routes:
            route_list.append({'route': [i], 'load': demands[i], 'saving': 0})

    total_saving = 0
    total_distance = 0
    all_routes = []
    data = []

    for idx, r in enumerate(route_list, 1):
        route = r['route']
        load = r['load']
        saving = r['saving']
        route_dist = dist[0][route[0]] + sum(dist[route[i]][route[i+1]] for i in range(len(route)-1)) + dist[route[-1]][0]
        route_names = ' - '.join([names[x] for x in route])
        data.append({
            "Vehicle": idx,
            "Route": f"Depot - {route_names} - Depot",
            "Load": load,
            "Saving": round(saving, 2),
            "Distance (km)": round(route_dist, 2)
        })
        total_saving += saving
        total_distance += route_dist
        all_routes.append([0] + route + [0])

    st.markdown("## Vehicle Routes")
    df_result = pd.DataFrame(data)
    st.dataframe(df_result, hide_index=True, use_container_width=True)

    st.markdown("## Summary")
    st.write(f"**Total Distance:** {total_distance:.2f} km")
    st.write(f"**Total Savings:** {total_saving:.2f}")
    draw_folium_map(all_routes, names, coords)

    return df_result, total_saving, total_distance

# --------------------- Streamlit Layout ---------------------

st.set_page_config(page_title="VRP Solver", layout="wide")

st.markdown("<h1 style='text-align:center;'>Clarke and Wright Savings Algorithm<br><small style='font-size:18px;'>Vehicle Routing Problem (No Time Windows)</small></h1>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    n_customers = st.number_input("Number of customers:", min_value=1, step=1)
    capacity = st.number_input("Vehicle capacity:", min_value=1, step=1)
    distance_mode = st.radio("Distance Matrix Mode", ["Manual", "Haversine"], index=1)

st.subheader("Depot Coordinates")
depot_lat = st.number_input("Latitude (use negative for South):")
depot_lon = st.number_input("Longitude (use negative for West):")
depot_coord = (depot_lat, depot_lon)

customers = []
for i in range(1, n_customers + 1):
    st.subheader(f"Customer {i}")
    name = st.text_input(f"Name", key=f"name_{i}")
    lat = st.number_input(f"Latitude (use negative for South)", key=f"lat_{i}")
    lon = st.number_input(f"Longitude (use negative for West)", key=f"lon_{i}")
    demand = st.number_input(f"Demand", min_value=1, step=1, key=f"demand_{i}")
    customers.append({"name": name, "coord": (lat, lon), "demand": demand})

coords = [depot_coord] + [c['coord'] for c in customers]
n = len(coords)
demands = [0] + [c['demand'] for c in customers]
names = ["Depot"] + [c['name'] for c in customers]

dist = [[0] * n for _ in range(n)]
if distance_mode == "Manual":
    st.subheader("Manual Distance Matrix")
    for i in range(n):
        for j in range(i + 1, n):
            val = st.number_input(f"Distance between {names[i]} and {names[j]}", min_value=0.0, key=f"d_{i}_{j}")
            dist[i][j] = dist[j][i] = val
else:
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(coords[i], coords[j])
            dist[i][j] = dist[j][i] = round(d, 2)

if st.button(" Solve VRP"):
    result_df, total_saving, total_distance = clarke_wright_vrp(names, coords, demands, dist, capacity)

    customer_df = pd.DataFrame(customers)
    customer_df.insert(0, "Customer", names[1:])

    # Combine both DataFrames into a single CSV output
    output = io.StringIO()
    output.write("Customer Information\n")
    customer_df.to_csv(output, index=False)
    output.write("\nVRP Results\n")
    result_df.to_csv(output, index=False)

    st.download_button(
        label="Download Report as CSV",
        data=output.getvalue(),
        file_name="vrp_report.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("<div style='text-align:center; color:gray;'>Made by <strong>joseph19</strong></div>", unsafe_allow_html=True)
