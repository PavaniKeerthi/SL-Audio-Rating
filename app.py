# app.py

import streamlit as st
from google.oauth2 import service_account
from google.cloud import storage
import gspread
import pandas as pd
import librosa
import numpy as np
import io
import tempfile
import os
from fpdf import FPDF
import openpyxl

# --- Page and App Configuration ---
st.set_page_config(layout="wide", page_title="Audio Quality Rater")

# --- Authentication and Client Functions (Unchanged) ---

@st.cache_resource
def get_gcs_client():
    """Initializes and returns a Google Cloud Storage client using Streamlit secrets."""
    try:
        creds_info = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(creds_info)
        return storage.Client(credentials=credentials)
    except Exception as e:
        st.error(f"GCS Auth Error: {e}")
        return None

@st.cache_resource
def get_gspread_client():
    """Initializes and returns a gspread client for Google Sheets."""
    try:
        creds_info = st.secrets["gcp_service_account"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        credentials = service_account.Credentials.from_service_account_info(
            creds_info, scopes=scopes
        )
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"Google Sheets Auth Error: {e}")
        return None

# --- Audio Analysis and GCS Functions (Unchanged) ---

def download_audio_from_gcs(gcs_url, client):
    """Downloads an audio file from a GCS URL into an in-memory stream."""
    try:
        if not gcs_url or not isinstance(gcs_url, str) or not gcs_url.startswith("https://storage.cloud.google.com/"):
            return None
        path_parts = gcs_url.replace("https://storage.cloud.google.com/", "").split('/')
        bucket_name = path_parts[0]
        blob_name = "/".join(path_parts[1:])
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if not blob.exists(): return None
        audio_data = blob.download_as_bytes()
        return io.BytesIO(audio_data)
    except Exception:
        return None

def analyze_audio_quality(file_path):
    """Analyzes audio quality with the 1-10 scoring model."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        if len(rms) > 0:
            if np.max(rms) > 0:
                signal_power = np.mean(rms**2)
                noise_power = np.min(rms*2) if np.min(rms*2) > 0 else 1e-10
                snr = 10 * np.log10(signal_power / noise_power)
            else: snr = 0
        else: snr = 0
        clipping_percentage = np.mean(np.abs(y) >= 0.99) * 100
        total_frames = len(rms)
        silence_percentage = (np.sum(rms < 0.01) / total_frames) * 100 if total_frames > 0 else 100
        score = 1.0
        if snr > 20: score += 4.0
        elif snr > 15: score += 3.0
        elif snr > 10: score += 2.0
        if clipping_percentage < 1: score += 3.0
        elif clipping_percentage < 5: score += 1.5
        if silence_percentage < 10: score += 2.0
        elif silence_percentage < 25: score += 1.0
        final_score = min(score, 10.0)
        return {"Overall Score": f"{final_score:.1f} / 10"}
    except Exception as e:
        return {"Error": str(e)}

# --- Report Generation Functions (Unchanged) ---

def dataframe_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Audio_Quality_Report')
    return output.getvalue()

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Audio Quality Analysis Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(df):
    pdf = PDF('P', 'mm', 'A4')
    pdf.add_page()
    # Methodology Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Rating Methodology', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5,
        "The overall audio quality score (from 1 to 10) is calculated based on several key technical metrics. "
        "A higher score indicates better audio quality.")
    pdf.ln(5)
    methodology = {
        "Signal-to-Noise Ratio (SNR):": "Measures the clarity of the speech compared to background noise.",
        "Clipping Percentage:": "Indicates distortion caused when the audio signal is too loud.",
        "Silence Percentage:": "Represents the amount of dead air or excessive pauses."
    }
    for title, desc in methodology.items():
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, title, 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(2)
    # Recommendations Section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '2. Recommendations for Better Ratings', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    recommendations = {
        "To Improve SNR:": "Use a high-quality microphone in a quiet environment.",
        "To Avoid Clipping:": "Monitor recording levels to keep them out of the 'red' zone.",
        "To Reduce Excessive Silence:": "Use audio editing software to trim long pauses."
    }
    for title, desc in recommendations.items():
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, title, 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(2)
    # Results Table Section
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '3. Analysis Results', 0, 1, 'L')
    pdf.set_font('Arial', 'B', 10)
    col_widths = [60, 60, 40]
    headers = list(df.columns)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, 'C')
    pdf.ln()
    pdf.set_font('Arial', '', 10)
    for _, row in df.iterrows():
        pdf.cell(col_widths[0], 8, str(row.iloc[0]), 1, 0, 'C')
        pdf.cell(col_widths[1], 8, str(row.iloc[1]), 1, 0, 'C')
        pdf.cell(col_widths[2], 8, str(row.iloc[2]), 1, 0, 'C')
        pdf.ln()
    return bytes(pdf.output(dest='S'))

# --- Main Application UI ---
st.title("📊 Simplified Audio Quality Rater")

st.markdown("""
Instructions:
1.  Ensure your Google Sheet is *shared* with the service account email.
2.  Ensure your sheet has a tab named exactly **URLs**.
3.  Ensure the URLs tab has columns named **Session Key** and **Status**.
4.  Paste the Google Sheet URL below.
5.  Select the desired status and click the button to start.
""")

gspread_client = get_gspread_client()
gcs_client = get_gcs_client()

if not gspread_client or not gcs_client:
    st.stop()

if 'results_df' not in st.session_state:
    st.session_state.results_df = None

sheet_url = st.text_input("Enter your Google Sheet URL:")

if sheet_url:
    try:
        spreadsheet = gspread_client.open_by_url(sheet_url)
        
        # --- Hardcoded Sheet and Column Names ---
        SHEET_NAME = "URLs"
        SESSION_KEY_COL = "Session Key"
        STATUS_COL = "Status"
        
        try:
            worksheet = spreadsheet.worksheet(SHEET_NAME)
        except gspread.exceptions.WorksheetNotFound:
            st.error(f"Error: A worksheet named '{SHEET_NAME}' was not found in your Google Sheet. Please check the name.")
            st.stop()
            
        df = pd.DataFrame(worksheet.get_all_records(head=1, default_blank=''))
        df = df.astype(str)

        if df.empty:
            st.warning(f"The sheet '{SHEET_NAME}' is empty or not formatted correctly.")
        else:
            # Check for required columns
            if SESSION_KEY_COL not in df.columns or STATUS_COL not in df.columns:
                st.error(f"Error: The '{SHEET_NAME}' sheet must contain columns named '{SESSION_KEY_COL}' and '{STATUS_COL}'.")
                st.stop()
            
            # --- Simplified Status Filter ---
            status_options = ["All", "Responded", "Not Responded"]
            selected_status = st.selectbox("Select the status to analyze:", options=status_options)

            # Filter the dataframe based on selection
            if selected_status == "All":
                df_to_process = df.copy()
            else:
                df_to_process = df[df[STATUS_COL] == selected_status].copy()

            if st.button("Calculate Average Audio Rating", type="primary"):
                if df_to_process.empty:
                    st.warning(f"No rows found with the status '{selected_status}'. Nothing to process.")
                    st.session_state.results_df = None
                else:
                    row_results = []
                    total_rows = len(df_to_process)
                    progress_bar = st.progress(0, text=f"Starting analysis on {total_rows} row(s)...")
                    
                    for i, (index, row) in enumerate(df_to_process.iterrows()):
                        session_key = row[SESSION_KEY_COL]
                        progress_text = f"Analyzing row for Session Key: {session_key} ({i + 1}/{total_rows})"
                        progress_bar.progress((i + 1) / total_rows, text=progress_text)
                        
                        row_scores = []
                        for cell_value in row:
                            if isinstance(cell_value, str) and cell_value.startswith("https://storage.cloud.google.com/"):
                                audio_stream = download_audio_from_gcs(cell_value, gcs_client)
                                if audio_stream:
                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
                                        temp_file.write(audio_stream.read())
                                        temp_file_path = temp_file.name
                                    try:
                                        rating = analyze_audio_quality(temp_file_path)
                                        if "Overall Score" in rating:
                                            score_str = rating["Overall Score"]
                                            numerical_score = float(score_str.split(' ')[0])
                                            row_scores.append(numerical_score)
                                    finally:
                                        if os.path.exists(temp_file_path):
                                            os.remove(temp_file_path)
                        
                        if row_scores:
                            average_score = sum(row_scores) / len(row_scores)
                            row_results.append({
                                "Session Key": session_key,
                                "Average Audio Score": f"{average_score:.1f} / 10",
                                "Audios Found": len(row_scores)
                            })
                        else:
                            row_results.append({
                                SESSION_KEY_COL: session_key,
                                "Average Audio Score": "N/A",
                                "Audios Found": 0
                            })
                    progress_bar.empty()
                    
                    if row_results:
                        st.session_state.results_df = pd.DataFrame(row_results)
                    else:
                        st.session_state.results_df = None

    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Spreadsheet not found. Please check the URL and that it's shared with the service account.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --- Display results and download buttons (remains the same) ---
if st.session_state.results_df is not None and not st.session_state.results_df.empty:
    st.subheader("Analysis Results")
    st.dataframe(st.session_state.results_df, use_container_width=True)

    st.subheader("Download Reports")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        excel_bytes = dataframe_to_excel_bytes(st.session_state.results_df)
        st.download_button(
            label="📥 Download Excel Report",
            data=excel_bytes,
            file_name="audio_quality_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="excel_download"
        )
    
    with dl_col2:
        pdf_bytes = create_pdf_report(st.session_state.results_df)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="audio_quality_report.pdf",
            mime="application/pdf",
            key="pdf_download"
        )