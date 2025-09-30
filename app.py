import streamlit as st
from datetime import datetime
import pandas as pd
from detection import detect_and_measure
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =============================
# CONFIG
# =============================
OWNER_PIN = "1234"  # 🔑 set your private PIN here

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("google_credentials.json", scope)
client = gspread.authorize(creds)
sheet_url = "https://docs.google.com/spreadsheets/d/1aLNdHM0QSQkgTsyaa0pFuoTCf7eFE7qmXpPSrlnFJO0/edit?usp=sharing"

# =============================
# OWNER MODE
# =============================
st.sidebar.header("🔒 Owner Mode")
pin_input = st.sidebar.text_input("Enter Owner PIN", type="password")
is_owner = (pin_input == OWNER_PIN)

if is_owner:
    st.sidebar.success("✅ Owner Mode Enabled")
else:
    st.sidebar.warning("Only owner can save to Google Sheets.")

# =============================
# MAIN APP
# =============================
st.title("📊 AQI / PM Dashboard with Secure Owner Mode")

def process(upload, label, sheet_name, is_hourly=False):
    if not upload:
        return
    from PIL import Image
    img = Image.open(upload).convert("RGB")
    values, annotated = detect_and_measure(img)
    st.subheader(label)
    if values is None:
        st.error("❌ Could not detect bars / axis ticks.")
        return

    st.image(annotated, use_column_width=True, caption=f"{label} — measured")

    # Create dataframe with Date/Time column
    if is_hourly:
        start_dt = st.datetime_input(f"⏰ Enter start time for {label}", value=datetime.now())
        times = pd.date_range(start=start_dt, periods=len(values), freq="H")
        df = pd.DataFrame({"DateTime": times, "Value": values})
    else:
        start_date = st.date_input(f"📅 Enter start date for {label}", value=datetime.today())
        df = pd.DataFrame({"Date": [start_date] * len(values), "Value": values})

    st.dataframe(df)

    # Save button (only in Owner Mode)
    if is_owner:
        if st.button(f"💾 Save {label} to Google Sheet"):
            try:
                sh = client.open_by_url(sheet_url)
                try:
                    worksheet = sh.worksheet(sheet_name)
                except gspread.exceptions.WorksheetNotFound:
                    worksheet = sh.add_worksheet(title=sheet_name, rows="1000", cols="2")
                    worksheet.append_row(df.columns.tolist())
                for _, row in df.iterrows():
                    worksheet.append_row(row.tolist())
                st.success(f"✅ Saved {label} data to Google Sheet ({sheet_name})")
            except Exception as e:
                st.error(f"Error saving: {e}")
    else:
        st.info("ℹ️ Login as owner to save this data.")

# =============================
# UPLOADERS
# =============================
col1, col2 = st.columns(2)
with col1:
    daily_aqi  = st.file_uploader("📂 Daily AQI",   ["jpg","jpeg","png"], key="d_aqi")
    hourly_aqi = st.file_uploader("📂 Hourly AQI",  ["jpg","jpeg","png"], key="h_aqi")
with col2:
    daily_pm   = st.file_uploader("📂 Daily PM2.5", ["jpg","jpeg","png"], key="d_pm")
    hourly_pm  = st.file_uploader("📂 Hourly PM2.5",["jpg","jpeg","png"], key="h_pm")

process(daily_aqi,  "Daily AQI",   "Aqi daily")
process(daily_pm,   "Daily PM2.5", "Pm daily")
process(hourly_aqi, "Hourly AQI",  "Aqi hourly", is_hourly=True)
process(hourly_pm,  "Hourly PM2.5","Pm hourly",  is_hourly=True)
