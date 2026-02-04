"""
Streamlit UI for 30-Day CLV Prediction.
Upload Excel file -> Call FastAPI -> Download CSV with predictions.
"""
import streamlit as st
import pandas as pd
import httpx
from io import BytesIO

API_BASE = "http://127.0.0.1:8000"  # FastAPI backend - try 8001 if 8000 fails


def _get_api_base():
    """Try 8001 first (fresh API), then 8000."""
    for port in [8001, 8000]:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    return "http://127.0.0.1:8000"  # default


def main():
    st.set_page_config(
        page_title="30-Day CLV Prediction",
        page_icon="🛒",
        layout="wide",
    )
    
    st.title("🛒 30-Day Customer Lifetime Value (CLV) Prediction")
    st.markdown(
        "Predict short-term customer spend using **Random Forest + Ridge/Lasso** "
        "and **RF + XGBoost + Ridge/Lasso** models."
    )
    
    api_base = _get_api_base()
    
    # Sidebar - API status
    with st.sidebar:
        st.header("API Status")
        try:
            r = httpx.get(f"{api_base}/health", timeout=2.0)
            status = r.json()
            if status.get("status") == "ok":
                st.success("✅ Backend connected")
            else:
                st.warning(f"⚠️ {status.get('error', 'Unknown')}")
        except Exception as e:
            st.error(f"❌ Backend offline")
            st.code("uvicorn api.main:app --reload --port 8000", language="bash")
            st.caption("Or port 8001 if 8000 is in use")
        
        st.divider()
        st.header("Dataset Format")
        st.markdown("""
        Upload **Online Retail II** format:
        - InvoiceNo, StockCode, Description
        - Quantity, InvoiceDate, UnitPrice
        - CustomerID, Country
        """)
        st.link_button("Download from Kaggle", "https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci")
    
    # Pipeline mode
    pipeline_mode = st.radio(
        "Pipeline",
        ["Full Pipeline (Upload → Clean → Load to DB → Predict)", "Quick Predict only"],
        help="Full: Clean uncleaned CSV, load into schema tables, store bad data in rejected_data with rejection_reason, then predict.",
    )
    use_full_pipeline = "Full Pipeline" in pipeline_mode
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Excel or CSV file (uncleaned for Full Pipeline)",
        type=["xlsx", "xls", "csv"],
        help="Transaction format: InvoiceNo, StockCode, Quantity, InvoiceDate, UnitPrice, CustomerID, Country"
    )
    
    if uploaded_file is not None:
        endpoint = f"{api_base}/upload-clean-predict" if use_full_pipeline else f"{api_base}/predict"
        with st.spinner("Processing... Full pipeline: cleaning, loading to DB, predicting. May take 1–2 min." if use_full_pipeline else "Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
                r = httpx.post(endpoint, files=files, timeout=300.0)
                r.raise_for_status()
                
                df = pd.read_csv(BytesIO(r.content))
                
                # Show stats for full pipeline
                if use_full_pipeline and r.headers.get("X-Original-Count"):
                    col_o, col_c, col_r = st.columns(3)
                    with col_o:
                        st.metric("Original rows", r.headers.get("X-Original-Count", "—"))
                    with col_c:
                        st.metric("Clean rows (loaded to DB)", r.headers.get("X-Clean-Count", "—"))
                    with col_r:
                        st.metric("Rejected rows", r.headers.get("X-Rejected-Count", "—"))
                    reasons = r.headers.get("X-Rejection-Reasons", "{}")
                    if reasons and reasons != "{}":
                        st.info(f"**Rejection reasons:** {reasons}")
                
                st.success(f"✅ Predictions ready for **{len(df)}** customers")
                
                # Model R2
                r2_1 = df["r2_model1"].iloc[0] if "r2_model1" in df.columns else "—"
                r2_2 = df["r2_model2"].iloc[0] if "r2_model2" in df.columns else "—"
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model 1 (RF+Ridge+Lasso) R²", f"{r2_1:.4f}" if isinstance(r2_1, (int, float)) else r2_1)
                with col2:
                    st.metric("Model 2 (RF+XGB+Ridge+Lasso) R²", f"{r2_2:.4f}" if isinstance(r2_2, (int, float)) else r2_2)
                
                # Preview table
                st.subheader("Predictions Preview")
                st.dataframe(df.head(20), use_container_width=True)
                
                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name="clv_predictions.csv",
                    mime="text/csv",
                )
                
            except httpx.HTTPStatusError as e:
                err = e.response.json() if e.response.headers.get("content-type", "").startswith("application/json") else {"detail": e.response.text}
                st.error(f"API Error: {err.get('detail', str(e))}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    else:
        st.info("👆 Upload an Excel or CSV file to get predictions")
        
        # Placeholder for sample output format
        st.subheader("Output Format")
        st.markdown("""
        The downloaded CSV will contain:
        - **customer_id**: Customer identifier
        - **r2_model1**: R² score for Model 1 (RF+Ridge+Lasso)
        - **r2_model2**: R² score for Model 2 (RF+XGBoost+Ridge+Lasso)
        - **predicted_30d_spend_model1**: Predicted 30-day spend (Model 1)
        - **predicted_30d_spend_model2**: Predicted 30-day spend (Model 2)
        - **predicted_no_of_products**: Estimated number of products
        """)


if __name__ == "__main__":
    main()
