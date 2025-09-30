import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
import json
import os

import gspread
from detection import detect_and_measure

st.set_page_config(page_title="AQI / PM Bar Reader", layout="wide")

# -----------------------------
# Google Sheets helpers
# -----------------------------
def _load_service_account_dict():
    """
    Try Streamlit secrets first, then fall back to local google_credentials.json
    when running in Codespaces/local.
    """
    if "gcp_service_account" in st.secrets:
        return dict(st.secrets["gcp_service_account"])
    # Fallback: local JSON file for dev
    if os.path.exists("google_credentials.json"):
        with open("google_credentials.json", "r", encoding="utf-8") as f:
            return json.load(f)
    raise RuntimeError("No service-account credentials found (secrets or google_credentials.json).")

def get_gsheet():
    sheet_id = st.secrets.get("SHEET_ID", "")
    if not sheet_id:
        raise RuntimeError("SHEET_ID missing in Streamlit secrets.")

    sa_dict = _load_service_account_dict()

    # Use gspread + google-auth (robust on Streamlit Cloud)
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    gc = gspread.service_account_from_dict(sa_dict, scopes=scopes)
    return gc.open_by_key(sheet_id)

def ensure_worksheet(book, name):
    try:
        return book.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
        return book.add_worksheet(title=name, rows="200", cols="50")

def write_to_sheet(sheet_name, rows, header):
    book = get_gsheet()
    ws = ensure_worksheet(book, sheet_name)
    ws.clear()
    ws.update([header] + rows)

# -----------------------------
# Date/Time generators
# -----------------------------
def generate_dates(values, start_date):
    return [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(values))]

def generate_hours(values, start_datetime):
    return [(start_datetime + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(len(values))]

# -----------------------------
# UI block
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

        # Save button (no PIN)
        if st.button(f"💾 Save {label} to Google Sheets"):
            try:
                if is_hourly:
                    rows = [[t, v] for t, v in zip(times, values)]
                    write_to_sheet(sheet_name, rows, ["Time & Date", "Value"])
                else:
                    rows = [[d, v] for d, v in zip(dates, values)]
                    write_to_sheet(sheet_name, rows, ["Date", "Value"])
                st.success(f"✅ Saved {len(values)} rows to {sheet_name}")
            except Exception as e:
                st.error(f"❌ Google Sheets write failed: {e}")

# -----------------------------
# Main UI
# -----------------------------
st.title("📊 AQI / PM Bar Reader Dashboard")
st.caption("Upload AQI / PM images ➝ detect bars ➝ align with axis labels ➝ save to Google Sheets")

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
