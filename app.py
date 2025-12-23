"""
Prism Worker Dashboard

A Streamlit dashboard to monitor and explore the Prism database.

Run:
    streamlit run dashboard/app.py
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# Page config
st.set_page_config(
    page_title="Prism Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_engine():
    """Create database connection."""
    db_url = os.getenv("DATABASE_URL", "")
    if not db_url:
        st.error("❌ DATABASE_URL not set!")
        st.info("Set it in your environment or .env file")
        st.stop()
    
    # Convert async URL to sync for pandas
    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    return create_engine(sync_url)


def run_query(query: str) -> pd.DataFrame:
    """Run a SQL query and return a DataFrame."""
    engine = get_db_engine()
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Query error: {e}")
        return pd.DataFrame()


def get_counts() -> dict:
    """Get counts of all main tables."""
    engine = get_db_engine()
    counts = {}
    tables = ["products", "retailers", "discovered_urls", "crawl_jobs", "product_prices", "product_images"]
    
    with engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar()
            except:
                counts[table] = 0
    
    return counts


# Sidebar
st.sidebar.title("🔮 Prism Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "🛍️ Products", "🖼️ Images", "🏪 Retailers", "🔗 Discovered URLs", "📋 Crawl Jobs", "🗑️ Clear Data"]
)

# Overview Page
if page == "📊 Overview":
    st.title("📊 Database Overview")
    
    counts = get_counts()
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Products", f"{counts.get('products', 0):,}")
    col2.metric("Retailers", f"{counts.get('retailers', 0):,}")
    col3.metric("Discovered URLs", f"{counts.get('discovered_urls', 0):,}")
    col4.metric("Crawl Jobs", f"{counts.get('crawl_jobs', 0):,}")
    
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Price Records", f"{counts.get('product_prices', 0):,}")
    col6.metric("Images", f"{counts.get('product_images', 0):,}")
    
    st.divider()
    
    # Recent Products
    st.subheader("🕐 Recent Products")
    recent_products = run_query("""
        SELECT p.title, p.price, p.currency, r.name as retailer, p.url, p.first_seen_at
        FROM products p
        LEFT JOIN retailers r ON p.retailer_id = r.id
        ORDER BY p.first_seen_at DESC
        LIMIT 10
    """)
    if not recent_products.empty:
        st.dataframe(recent_products, use_container_width=True)
    else:
        st.info("No products yet")
    
    # Recent Jobs
    st.subheader("📋 Recent Crawl Jobs")
    recent_jobs = run_query("""
        SELECT id, job_type, status, base_url, 
               products_found, urls_discovered,
               started_at, completed_at
        FROM crawl_jobs
        ORDER BY created_at DESC
        LIMIT 10
    """)
    if not recent_jobs.empty:
        st.dataframe(recent_jobs, use_container_width=True)
    else:
        st.info("No crawl jobs yet")


# Products Page
elif page == "🛍️ Products":
    st.title("🛍️ Products")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("🔍 Search title", "")
    with col2:
        retailers = run_query("SELECT id, name FROM retailers ORDER BY name")
        if not retailers.empty:
            retailer_options = ["All"] + retailers['name'].tolist()
            selected_retailer = st.selectbox("🏪 Retailer", retailer_options)
        else:
            selected_retailer = "All"
    with col3:
        limit = st.number_input("Limit", min_value=10, max_value=1000, value=100)
    
    # Build query
    query = """
        SELECT 
            p.id,
            p.title,
            p.retailer_sku as sku,
            p.price,
            p.currency,
            p.brand,
            p.in_stock,
            r.name as retailer,
            p.url,
            p.first_seen_at,
            p.last_updated_at
        FROM products p
        LEFT JOIN retailers r ON p.retailer_id = r.id
        WHERE 1=1
    """
    
    if search:
        query += f" AND p.title ILIKE '%{search}%'"
    if selected_retailer != "All":
        query += f" AND r.name = '{selected_retailer}'"
    
    query += f" ORDER BY p.first_seen_at DESC LIMIT {limit}"
    
    products = run_query(query)
    
    st.metric("Total Results", len(products))
    
    if not products.empty:
        st.dataframe(products, use_container_width=True, height=600)
        
        # Download button
        csv = products.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "products.csv", "text/csv")
    else:
        st.info("No products found")


# Images Page
elif page == "🖼️ Images":
    st.title("🖼️ Product Images")
    
    # Get image count
    img_count = run_query("SELECT COUNT(*) as count FROM product_images")
    total_images = int(img_count['count'].iloc[0]) if not img_count.empty else 0
    
    st.metric("Total Images", f"{total_images:,}")
    
    # Limit selector
    limit = st.slider("Number of images to show", 10, 100, 30)
    
    # Fetch images with product info
    images = run_query(f"""
        SELECT 
            pi.source_url,
            pi.stored_url,
            pi.width,
            pi.height,
            pi.alt_text,
            p.title as product_title,
            p.url as product_url
        FROM product_images pi
        LEFT JOIN products p ON pi.product_id = p.id
        ORDER BY pi.created_at DESC
        LIMIT {limit}
    """)
    
    if not images.empty:
        # Display images in a grid
        cols_per_row = 6
        
        for i in range(0, len(images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(images):
                    row = images.iloc[idx]
                    img_url = row['stored_url'] or row['source_url']
                    with col:
                        try:
                            st.image(img_url, width=100, caption=row['product_title'][:20] + "..." if row['product_title'] and len(row['product_title']) > 20 else row['product_title'])
                        except:
                            st.text("🖼️ Error")
    else:
        st.info("No images yet")


# Retailers Page
elif page == "🏪 Retailers":
    st.title("🏪 Retailers")
    
    retailers = run_query("""
        SELECT 
            r.id,
            r.name,
            r.domain,
            r.platform,
            r.crawl_enabled,
            r.last_crawled_at,
            r.created_at,
            COUNT(DISTINCT p.id) as product_count
        FROM retailers r
        LEFT JOIN products p ON r.id = p.retailer_id
        GROUP BY r.id, r.name, r.domain, r.platform, r.crawl_enabled, r.last_crawled_at, r.created_at
        ORDER BY product_count DESC
    """)
    
    if not retailers.empty:
        st.dataframe(retailers, use_container_width=True)
    else:
        st.info("No retailers yet")


# Discovered URLs Page
elif page == "🔗 Discovered URLs":
    st.title("🔗 Discovered URLs")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        url_type = st.selectbox("Type", ["All", "product", "category", "other"])
    with col2:
        extracted = st.selectbox("Extracted?", ["All", "Yes", "No"])
    with col3:
        limit = st.number_input("Limit", min_value=10, max_value=1000, value=100, key="url_limit")
    
    query = """
        SELECT 
            du.url,
            du.url_type,
            du.source,
            du.extracted,
            du.visited,
            du.discovered_at
        FROM discovered_urls du
        WHERE 1=1
    """
    
    if url_type != "All":
        query += f" AND du.url_type = '{url_type}'"
    if extracted == "Yes":
        query += " AND du.extracted = true"
    elif extracted == "No":
        query += " AND du.extracted = false"
    
    query += f" ORDER BY du.discovered_at DESC LIMIT {limit}"
    
    urls = run_query(query)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    stats = run_query("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN extracted THEN 1 ELSE 0 END) as extracted_count,
            SUM(CASE WHEN url_type = 'product' THEN 1 ELSE 0 END) as products
        FROM discovered_urls
    """)
    if not stats.empty and stats['total'].iloc[0] is not None:
        col1.metric("Total URLs", f"{int(stats['total'].iloc[0] or 0):,}")
        col2.metric("Extracted", f"{int(stats['extracted_count'].iloc[0] or 0):,}")
        col3.metric("Product URLs", f"{int(stats['products'].iloc[0] or 0):,}")
    
    if not urls.empty:
        st.dataframe(urls, use_container_width=True, height=500)
    else:
        st.info("No discovered URLs yet")


# Crawl Jobs Page
elif page == "📋 Crawl Jobs":
    st.title("📋 Crawl Jobs")
    
    jobs = run_query("""
        SELECT 
            id,
            job_type,
            status,
            base_url,
            urls_discovered,
            urls_crawled,
            products_found,
            products_updated,
            errors,
            error_message,
            created_at,
            started_at,
            completed_at,
            EXTRACT(EPOCH FROM (completed_at - started_at)) as duration_seconds
        FROM crawl_jobs
        ORDER BY created_at DESC
        LIMIT 100
    """)
    
    if not jobs.empty:
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Jobs", len(jobs))
        col2.metric("Completed", len(jobs[jobs['status'] == 'completed']))
        col3.metric("Failed", len(jobs[jobs['status'] == 'failed']))
        col4.metric("Products Found", int(jobs['products_found'].sum()))
        
        st.dataframe(jobs, use_container_width=True, height=500)
    else:
        st.info("No crawl jobs yet")


# Clear Data Page
elif page == "🗑️ Clear Data":
    st.title("🗑️ Clear Data")
    
    st.warning("⚠️ This will permanently delete data from the database!")
    
    counts = get_counts()
    
    st.subheader("Current Data")
    for table, count in counts.items():
        st.text(f"{table}: {count:,} rows")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Clear Specific Tables")
        tables_to_clear = st.multiselect(
            "Select tables to clear",
            ["products", "product_prices", "product_images", "discovered_urls", "crawl_jobs", "retailers"]
        )
        
        if tables_to_clear:
            confirm = st.checkbox(f"I confirm I want to delete data from: {', '.join(tables_to_clear)}")
            if confirm and st.button("🗑️ Clear Selected Tables", type="primary"):
                engine = get_db_engine()
                with engine.connect() as conn:
                    for table in tables_to_clear:
                        conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                    conn.commit()
                st.success(f"✅ Cleared: {', '.join(tables_to_clear)}")
                st.rerun()
    
    with col2:
        st.subheader("Clear All Data")
        st.error("🚨 Nuclear option - clears everything!")
        
        confirm_all = st.checkbox("I understand this will delete ALL data")
        if confirm_all and st.button("☢️ Clear All Tables", type="secondary"):
            engine = get_db_engine()
            with engine.connect() as conn:
                # Order matters due to foreign keys
                tables = [
                    "product_images",
                    "product_prices", 
                    "products",
                    "discovered_urls",
                    "crawl_jobs",
                    "retailers"
                ]
                for table in tables:
                    try:
                        conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
                    except Exception as e:
                        st.warning(f"Could not clear {table}: {e}")
                conn.commit()
            st.success("✅ All tables cleared!")
            st.rerun()


# Footer
st.sidebar.divider()
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
