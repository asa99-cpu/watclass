import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from detection import detect_and_measure

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -----------------------------
# Owner mode (PIN)
# -----------------------------
st.set_page_config(page_title="AQI / PM Bar Reader", layout="wide")

if "owner_ok" not in st.session_state:
    st.session_state["owner_ok"] = False

with st.sidebar:
    st.header("🔐 Owner Mode")
    pin_input = st.text_input("Enter Owner PIN", type="password")
    if st.button("Enable Owner Mode"):
        expected = st.secrets.get("OWNER_PIN", "")
        if pin_input and pin_input == expected:
            st.session_state["owner_ok"] = True
            st.success("Owner mode enabled")
        else:
            st.error("Wrong PIN")
    st.caption("Only the owner can save to Google Sheets.")

# -----------------------------
# Google Sheets Setup (from secrets)
# -----------------------------
SHEET_ID = st.secrets.get("SHEET_ID", "")
GCP_SA = st.secrets.get("gcp_service_account", None)

def get_gsheet():
    if not SHEET_ID or not GCP_SA:
        raise RuntimeError("SHEET_ID or gcp_service_account missing in Streamlit secrets.")
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(GCP_SA, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID)

def ensure_worksheet(book, name):
    try:
        ws = book.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
        ws = book.add_worksheet(title=name, rows="200", cols="50")
    return ws

def write_to_sheet(sheet_name, rows, header):
    book = get_gsheet()
    ws = ensure_worksheet(book, sheet_name)
    ws.clear()
    ws.update([header] + rows)

# -----------------------------
# Date/Time Generators
# -----------------------------
def generate_dates(values, start_date):
    return [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(values))]

def generate_hours(values, start_datetime):
    return [(start_datetime + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(len(values))]

# -----------------------------
# Processing UI block (side-by-side)
# -----------------------------
def process(upload, label, sheet_name, is_hourly=False):
    col_upload, col_result = st.columns([1,2])

    with col_upload:
        st.subheader(label)
        st.write("📤 Upload image here")

    with col_result:
        if not upload:
            st.info("⬆️ Please upload an image to see results")
            return

        img = Image.open(upload).convert("RGB")
        values, annotated = detect_and_measure(img)

        if values is None:
            st.error("❌ Couldn’t detect bars / axis ticks.")
            return

        st.image(annotated, use_column_width=True, caption=f"{label}")

        if not is_hourly:
            start_date = st.date_input(
                f"📅 Start date for {label}",
                value=datetime.today().date(),
                key=f"date_{sheet_name}"
            )
            dates = generate_dates(values, datetime.combine(start_date, datetime.min.time()))
            df = pd.DataFrame({"Date": dates, "Value": values})
        else:
            start_date = st.date_input(
                f"📅 Start date for {label}",
                value=datetime.today().date(),
                key=f"date_{sheet_name}"
            )
            start_time = st.time_input(
                f"⏰ Start time for {label}",
                value=datetime.now().time(),
                key=f"time_{sheet_name}"
            )
            start_dt = datetime.combine(start_date, start_time)
            times = generate_hours(values, start_dt)
            df = pd.DataFrame({"Time & Date": times, "Value": values})

        st.dataframe(df, use_container_width=True)

        # Save button ONLY for owner
        if st.session_state["owner_ok"]:
            if st.button(f"💾 Save {label} to Google Sheets"):
                if is_hourly:
                    rows = [[t, v] for t, v in zip(times, values)]
                    write_to_sheet(sheet_name, rows, ["Time & Date", "Value"])
                else:
                    rows = [[d, v] for d, v in zip(dates, values)]
                    write_to_sheet(sheet_name, rows, ["Date", "Value"])
                st.success(f"✅ Saved {len(values)} rows to {sheet_name}")
        else:
            st.info("Saving is disabled for public users. Owner: enable with PIN in sidebar.")

# -----------------------------
# Main UI
# -----------------------------
st.title("📊 AQI / PM Bar Reader Dashboard")
st.caption("Upload AQI / PM images ➝ detect bars ➝ align with axis labels ➝ (Owner-only) save results to Google Sheets")

st.markdown("---")
daily_aqi  = st.file_uploader("📂 Upload Daily AQI",   ["jpg","jpeg","png"], key="d_aqi")
process(daily_aqi, "Daily AQI", "Aqi daily", is_hourly=False)

st.markdown("---")
hourly_aqi = st.file_uploader("📂 Upload Hourly AQI",  ["jpg","jpeg","png"], key="h_aqi")
process(hourly_aqi, "Hourly AQI", "Aqi hourly", is_hourly=True)

st.markdown("---")
daily_pm   = st.file_uploader("📂 Upload Daily PM2.5", ["jpg","jpeg","png"], key="d_pm")
process(daily_pm, "Daily PM2.5", "Pm daily", is_hourly=False)

st.markdown("---")
hourly_pm  = st.file_uploader("📂 Upload Hourly PM2.5",["jpg","jpeg","png"], key="h_pm")
process(hourly_pm, "Hourly PM2.5", "Pm hourly", is_hourly=True)
 
