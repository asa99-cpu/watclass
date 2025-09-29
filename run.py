import streamlit as st
import pandas as pd
import altair as alt

# Title
st.title("Kurdistan Weather Data Explorer")

# Load CSV
csv_path = "/workspaces/watclass/data.csv"
df = pd.read_csv(csv_path)

# Sidebar filters
st.sidebar.header("Filters")
min_temp, max_temp = int(df["Temperature"].min()), int(df["Temperature"].max())
temp_range = st.sidebar.slider("Temperature Range", min_temp, max_temp, (min_temp, max_temp))

min_hum, max_hum = int(df["Humidity"].min()), int(df["Humidity"].max())
hum_range = st.sidebar.slider("Humidity Range", min_hum, max_hum, (min_hum, max_hum))

# Filter data
filtered_df = df[(df["Temperature"] >= temp_range[0]) & (df["Temperature"] <= temp_range[1]) &
                 (df["Humidity"] >= hum_range[0]) & (df["Humidity"] <= hum_range[1])]

# Show filtered table
st.subheader("Filtered Data")
st.dataframe(filtered_df)

# Summary statistics
st.subheader("Summary Statistics")
st.write(filtered_df.describe())

# Charts
st.subheader("Temperature vs Humidity")
chart = alt.Chart(filtered_df).mark_circle(size=100).encode(
    x='Temperature',
    y='Humidity',
    tooltip=['City', 'Temperature', 'Humidity']
).interactive()

st.altair_chart(chart, use_container_width=True)

# Histogram for Temperature
st.subheader("Temperature Distribution")
temp_hist = alt.Chart(filtered_df).mark_bar().encode(
    alt.X("Temperature:Q", bin=alt.Bin(maxbins=20)),
    y='count()',
    tooltip=['count()']
)
st.altair_chart(temp_hist, use_container_width=True)

# Histogram for Humidity
st.subheader("Humidity Distribution")
hum_hist = alt.Chart(filtered_df).mark_bar().encode(
    alt.X("Humidity:Q", bin=alt.Bin(maxbins=20)),
    y='count()',
    tooltip=['count()']
)
st.altair_chart(hum_hist, use_container_width=True)
