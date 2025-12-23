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

# Custom CSS - Compact metrics
st.markdown("""
<style>
    /* Remove card styling, make metrics compact */
    div[data-testid="stMetric"] {
        background: none !important;
        padding: 0 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    /* Compact the metric columns */
    div[data-testid="column"] {
        padding: 0 0.5rem !important;
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
    ["📊 Overview", "🛍️ Products", "🖼️ Images", "💰 Price History", "🏪 Retailers", "🔗 Discovered URLs", "📋 Crawl Jobs", "🗑️ Clear Data"]
)

# Overview Page
if page == "📊 Overview":
    st.title("📊 Database Overview")
    
    counts = get_counts()
    
    # All metrics in one row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Products", f"{counts.get('products', 0):,}")
    col2.metric("Retailers", f"{counts.get('retailers', 0):,}")
    col3.metric("URLs", f"{counts.get('discovered_urls', 0):,}")
    col4.metric("Jobs", f"{counts.get('crawl_jobs', 0):,}")
    col5.metric("Prices", f"{counts.get('product_prices', 0):,}")
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
    
    # Build query - ALL fields
    query = """
        SELECT 
            p.id,
            p.title,
            p.retailer_sku as sku,
            r.name as retailer,
            p.brand,
            p.price,
            p.original_price,
            p.currency,
            p.in_stock,
            p.stock_level,
            p.rating,
            p.review_count,
            p.categories,
            p.tags,
            p.upc,
            p.gtin,
            p.mpn,
            p.description,
            p.url,
            p.canonical_url,
            p.is_active,
            p.first_seen_at,
            p.last_updated_at,
            p.last_crawled_at
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
        # Don't use use_container_width=True to enable horizontal scrolling
        st.dataframe(products, height=600)
        
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
            pi.id,
            pi.product_id,
            pi.source_url,
            pi.stored_url,
            pi.position,
            pi.alt_text,
            pi.width,
            pi.height,
            pi.image_hash,
            pi.created_at,
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


# Price History Page
elif page == "💰 Price History":
    st.title("💰 Price History")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("Limit", min_value=10, max_value=1000, value=100, key="price_limit")
    with col2:
        # Search by product title
        search_product = st.text_input("Search product (title)", "")
    
    if search_product:
        prices = run_query(f"""
            SELECT 
                pp.product_id,
                p.title as product_title,
                pp.price,
                pp.original_price,
                pp.currency,
                pp.in_stock,
                pp.recorded_at
            FROM product_prices pp
            LEFT JOIN products p ON pp.product_id = p.id
            WHERE LOWER(p.title) LIKE LOWER('%{search_product}%')
            ORDER BY pp.recorded_at DESC
            LIMIT {limit}
        """)
    else:
        prices = run_query(f"""
            SELECT 
                pp.product_id,
                p.title as product_title,
                pp.price,
                pp.original_price,
                pp.currency,
                pp.in_stock,
                pp.recorded_at
            FROM product_prices pp
            LEFT JOIN products p ON pp.product_id = p.id
            ORDER BY pp.recorded_at DESC
            LIMIT {limit}
        """)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    total_prices = run_query("SELECT COUNT(*) as count FROM product_prices")
    if not total_prices.empty:
        col1.metric("Total Price Records", f"{int(total_prices['count'].iloc[0] or 0):,}")
    
    if not prices.empty:
        col2.metric("Avg Price", f"${prices['price'].mean():.2f}")
        col3.metric("Products with Discounts", len(prices[prices['original_price'].notna() & (prices['original_price'] > prices['price'])]))
        
        st.dataframe(prices, height=500)
        
        # Download button
        csv = prices.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "price_history.csv", "text/csv")
    else:
        st.info("No price history yet")


# Retailers Page
elif page == "🏪 Retailers":
    st.title("🏪 Retailers")
    
    retailers = run_query("""
        SELECT 
            r.id,
            r.slug,
            r.name,
            r.domain,
            r.platform,
            r.crawl_enabled,
            r.crawl_frequency_hours,
            r.last_crawled_at,
            r.created_at,
            r.updated_at,
            COUNT(DISTINCT p.id) as product_count
        FROM retailers r
        LEFT JOIN products p ON r.id = p.retailer_id
        GROUP BY r.id, r.slug, r.name, r.domain, r.platform, r.crawl_enabled, 
                 r.crawl_frequency_hours, r.last_crawled_at, r.created_at, r.updated_at
        ORDER BY product_count DESC
    """)
    
    if not retailers.empty:
        st.dataframe(retailers, height=400)
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
            du.id,
            du.job_id,
            du.url,
            du.url_type,
            du.source,
            du.depth,
            du.visited,
            du.extracted,
            du.product_id,
            du.discovered_at,
            du.visited_at
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
        st.dataframe(urls, height=500)
    else:
        st.info("No discovered URLs yet")


# Crawl Jobs Page
elif page == "📋 Crawl Jobs":
    st.title("📋 Crawl Jobs")
    
    jobs = run_query("""
        SELECT 
            cj.id,
            cj.retailer_id,
            r.name as retailer_name,
            cj.base_url,
            cj.job_type,
            cj.status,
            cj.urls_discovered,
            cj.urls_crawled,
            cj.products_found,
            cj.products_updated,
            cj.errors,
            cj.error_message,
            cj.created_at,
            cj.started_at,
            cj.completed_at,
            EXTRACT(EPOCH FROM (cj.completed_at - cj.started_at)) as duration_seconds
        FROM crawl_jobs cj
        LEFT JOIN retailers r ON cj.retailer_id = r.id
        ORDER BY cj.created_at DESC
        LIMIT 100
    """)
    
    if not jobs.empty:
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Jobs", len(jobs))
        col2.metric("Completed", len(jobs[jobs['status'] == 'completed']))
        col3.metric("Failed", len(jobs[jobs['status'] == 'failed']))
        col4.metric("Products Found", int(jobs['products_found'].sum()))
        
        st.dataframe(jobs, height=500)
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
