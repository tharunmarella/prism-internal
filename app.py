"""
Prism Worker Dashboard

A Streamlit dashboard to monitor and explore the Prism database.

Run:
    streamlit run dashboard/app.py
    
Environment Variables:
    DATABASE_URL: PostgreSQL connection URL
    REDIS_URL: Redis connection URL  
    PRISM_API_URL: Prism API base URL
"""

import os
import sys
import json
import logging
from datetime import datetime

import pandas as pd
import redis
import streamlit as st

# =============================================================================
# Configure Logging - Output to stdout for container visibility
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("prism-dashboard")

# Log startup
logger.info("=" * 60)
logger.info("Prism Dashboard starting...")
logger.info(f"DATABASE_URL set: {bool(os.getenv('DATABASE_URL'))}")
logger.info(f"REDIS_URL set: {bool(os.getenv('REDIS_URL'))}")
logger.info(f"PRISM_API_URL: {os.getenv('PRISM_API_URL', 'https://prism-api-production.up.railway.app')}")
logger.info("=" * 60)

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
    
    /* Force horizontal scrollbar on dataframes */
    div[data-testid="stDataFrame"] > div {
        overflow-x: auto !important;
    }
    div[data-testid="stDataFrame"] iframe {
        min-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


def get_db_connection():
    """
    Get database connection using Streamlit's built-in st.connection.
    
    This handles:
    - Connection pooling and management
    - Automatic retries on connection failures
    - Query result caching
    - Secrets management
    """
    db_url = os.getenv("DATABASE_URL", "")
    
    if not db_url:
        logger.error("DATABASE_URL environment variable not set!")
        st.error("❌ DATABASE_URL not set!")
        st.info("Set it in your environment or .env file")
        st.stop()
    
    # Convert async URL to sync for SQLAlchemy
    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    # Also handle postgres:// scheme
    if sync_url.startswith("postgres://"):
        sync_url = sync_url.replace("postgres://", "postgresql://", 1)
    
    # Mask password for logging
    masked_url = sync_url.split("@")[-1] if "@" in sync_url else sync_url
    logger.info(f"Connecting to database: {masked_url}")
    
    # Use st.connection with the URL - handles retries, pooling, caching
    return st.connection("postgresql", type="sql", url=sync_url)


@st.cache_resource
def get_redis_client():
    """Create Redis connection."""
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        logger.warning("REDIS_URL not set, Redis features will be disabled")
        return None
    
    try:
        logger.info("Creating Redis client...")
        client = redis.from_url(redis_url, decode_responses=True)
        # Test connection
        client.ping()
        logger.info("Redis connection successful")
        return client
    except Exception as e:
        logger.exception(f"Failed to connect to Redis: {e}")
        return None


def get_queue_stats():
    """Get stats for all queues via prism-api."""
    import requests
    
    api_url = os.getenv("PRISM_API_URL", "https://prism-api-production.up.railway.app")
    logger.debug(f"Fetching queue stats from {api_url}/queues/stats")
    
    try:
        resp = requests.get(f"{api_url}/queues/stats", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            logger.debug(f"Queue stats received: {len(data.get('queues', {}))} queues")
            return data.get("queues", {})
        else:
            logger.warning(f"Queue stats API returned status {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Failed to fetch queue stats from API: {e}")
    
    # Fallback to empty stats
    return {q: {"pending": 0, "failed": 0} for q in ["orchestrate", "platform", "discover", "extract", "index", "price"]}


def get_dlq_jobs():
    """Get all jobs from the native RabbitMQ Dead Letter Queue."""
    import requests
    from urllib.parse import urlparse
    
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://prism:prism_dev_password@localhost:5672/")
    parsed = urlparse(rabbitmq_url)
    
    # RabbitMQ Management API
    user = parsed.username or "prism"
    password = parsed.password or "prism_dev_password"
    host = parsed.hostname or "localhost"
    vhost = parsed.path.lstrip('/') or "%2f"
    
    api_url = f"http://{host}:15672/api/queues/{vhost}/prism.dead_letter/get"
        
        try:
        # Note: We use ackmode: ack_requeue_true to "peek" at the messages
        resp = requests.post(
            api_url,
            auth=(user, password),
            json={"count": 50, "ackmode": "ack_requeue_true", "encoding": "auto", "truncate": 50000},
            timeout=5
        )
        if resp.status_code == 200:
            messages = resp.json()
            jobs = []
            for msg in messages:
                try:
                    payload = json.loads(msg["payload"])
                    # Extract info from native RabbitMQ payload
                    job = {
                        "id": msg.get("properties", {}).get("message_id", "N/A"),
                        "task": payload.get("task", "unknown"),
                        "args": payload.get("args", []),
                        "kwargs": payload.get("kwargs", {}),
                        "timestamp": payload.get("timestamp"),
                        "failed_at": datetime.now(), 
                        "routing_key": msg.get("routing_key"),
                        "exchange": msg.get("exchange")
                    }
                    jobs.append(job)
                except Exception as e:
                    logger.debug(f"Failed to parse RabbitMQ message: {e}")
                    continue
            return jobs
        except Exception as e:
        logger.warning(f"Failed to fetch DLQ from RabbitMQ Management API: {e}")
    
    return []


def clear_dlq() -> int:
    """Purge the RabbitMQ Dead Letter Queue."""
    import requests
    from urllib.parse import urlparse
    
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://prism:prism_dev_password@localhost:5672/")
    parsed = urlparse(rabbitmq_url)
    
    user = parsed.username or "prism"
    password = parsed.password or "prism_dev_password"
    host = parsed.hostname or "localhost"
    vhost = parsed.path.lstrip('/') or "%2f"
    
    api_url = f"http://{host}:15672/api/queues/{vhost}/prism.dead_letter/contents"
    
    try:
        resp = requests.delete(
            api_url,
            auth=(user, password),
            timeout=5
        )
        if resp.status_code == 204:
            return 1 # Success
    except Exception as e:
        logger.error(f"Failed to clear RabbitMQ DLQ: {e}")
    
    return 0


def clear_queue(queue_name: str) -> int:
    """Purge a specific RabbitMQ queue."""
    import requests
    from urllib.parse import urlparse
    
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://prism:prism_dev_password@localhost:5672/")
    parsed = urlparse(rabbitmq_url)
    
    user = parsed.username or "prism"
    password = parsed.password or "prism_dev_password"
    host = parsed.hostname or "localhost"
    vhost = parsed.path.lstrip('/') or "%2f"
    
    api_url = f"http://{host}:15672/api/queues/{vhost}/{queue_name}/contents"
    
    try:
        resp = requests.delete(
            api_url,
            auth=(user, password),
            timeout=5
        )
        if resp.status_code == 204:
            return 1
    except Exception as e:
        logger.error(f"Failed to clear RabbitMQ queue {queue_name}: {e}")
    
    return 0


def clear_all_queues() -> dict:
    """Purge all RabbitMQ queues."""
    queues = ["orchestrate", "platform", "discover", "extract", "index", "price"]
    results = {}
    for queue in queues:
        results[queue] = clear_queue(queue)
    return results


def get_active_jobs():
    """Get active jobs with progress via prism-api."""
    import requests
    
    api_url = os.getenv("PRISM_API_URL", "https://prism-api-production.up.railway.app")
    logger.debug(f"Fetching active jobs from {api_url}/queues/active-jobs")
    
    try:
        resp = requests.get(f"{api_url}/queues/active-jobs", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            jobs = data.get("jobs", [])
            logger.debug(f"Got {len(jobs)} active jobs")
            return jobs
        else:
            logger.warning(f"Active jobs API returned status {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to fetch active jobs from API: {e}")
    
    return []


def get_redis_info():
    """Get Redis server info."""
    r = get_redis_client()
    if not r:
        logger.debug("No Redis client available for info")
        return {}
    
    try:
        info = r.info()
        result = {
            "memory": info.get("used_memory_human", "N/A"),
            "clients": info.get("connected_clients", 0),
            "keys": r.dbsize(),
            "uptime": info.get("uptime_in_days", 0)
        }
        logger.debug(f"Redis info: {result}")
        return result
    except Exception as e:
        logger.warning(f"Failed to get Redis info: {e}")
        return {}


def run_query(query: str, ttl: str = "1m") -> pd.DataFrame:
    """
    Run a SQL query and return a DataFrame.
    
    Uses st.connection which handles:
    - Connection management and retries
    - Query result caching (default 1 minute TTL)
    """
    # Log first 100 chars of query for debugging
    query_preview = query.replace('\n', ' ')[:100]
    logger.debug(f"Executing query: {query_preview}...")
    
    try:
        conn = get_db_connection()
        result = conn.query(query, ttl=ttl)
        logger.debug(f"Query returned {len(result)} rows")
        return result
    except Exception as e:
        logger.exception(f"Query failed: {e}\nQuery: {query[:500]}")
        st.error(f"Query error: {e}")
        return pd.DataFrame()


def get_counts() -> dict:
    """Get counts of all main tables using st.connection."""
    logger.info("Fetching table counts...")
    counts = {}
    tables = ["products", "retailers", "discovered_urls", "crawl_jobs", "product_prices", "product_images"]
    
    try:
        conn = get_db_connection()
        for table in tables:
            try:
                # Use conn.query with short TTL for counts
                result = conn.query(f"SELECT COUNT(*) as count FROM {table}", ttl="30s")
                counts[table] = int(result['count'].iloc[0]) if not result.empty else 0
                logger.debug(f"  {table}: {counts[table]} rows")
            except Exception as e:
                logger.error(f"Failed to count table {table}: {e}")
                counts[table] = 0
    except Exception as e:
        logger.exception(f"Failed to connect to database for counts: {e}")
        # Return zeros for all tables
        counts = {table: 0 for table in tables}
    
    logger.info(f"Table counts: {counts}")
    return counts


# Sidebar
st.sidebar.title("🔮 Prism Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "📤 Batch Scrape", "🔍 Semantic Search", "📬 Queues", "🛍️ Products", "🏪 Store", "🖼️ Images", "💰 Price History", "🏪 Retailers", "🔗 Discovered URLs", "📋 Crawl Jobs", "🗑️ Clear Data"]
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
        st.caption(f"Fields: {', '.join(recent_products.columns)}")
        st.dataframe(recent_products, use_container_width=True)
    else:
        st.info("No products yet")

# Batch Scrape Page
elif page == "📤 Batch Scrape":
    st.title("📤 Batch Scrape")
    st.caption("Upload a CSV file with store domains to scrape")
    
    # Check connections
    r = get_redis_client()
    if not r:
        st.error("❌ REDIS_URL not set! Cannot track batch progress.")
        st.stop()
    
    # Tab layout: Upload vs Monitor
    tab1, tab2 = st.tabs(["📤 Upload CSV", "📊 Monitor Batches"])
    
    with tab1:
        st.subheader("Upload Store List")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="CSV should have a column with domain names (e.g., 'domain', 'current_domain', or 'url')"
        )
        
        if uploaded_file is not None:
            try:
                import io
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Try to find domain column
                domain_col = None
                for col in ["current_domain", "domain", "original_domain", "url"]:
                    if col in df.columns:
                        domain_col = col
                        break
                
                if not domain_col:
                    st.error(f"❌ No domain column found. Available columns: {list(df.columns)}")
                else:
                    st.info(f"Using column: **{domain_col}**")
                    
                    # Preview
                    st.write("**Preview (first 10 rows):**")
                    st.dataframe(df.head(10))
                    
                    # Batch settings
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        batch_name = st.text_input("Batch Name", value="My Batch")
                    with col2:
                        concurrency = st.number_input("Concurrency", min_value=1, max_value=10, value=1, help="How many stores to scrape in parallel")
                    with col3:
                        limit = st.number_input("Limit", min_value=1, max_value=len(df), value=min(len(df), 100), help="Max stores to process")
                    
                    # Start button
                    if st.button("🚀 Start Batch Scrape", type="primary"):
                        # Extract domains
                        domains = df[domain_col].dropna().astype(str).tolist()[:limit]
                        
                        # Clean domains
                        cleaned_domains = []
                        for d in domains:
                            d = d.strip().replace("https://", "").replace("http://", "").rstrip("/")
                            if d:
                                cleaned_domains.append(d)
                        
                        if not cleaned_domains:
                            st.error("No valid domains found!")
                        else:
                            # Create batch in Redis
                            import uuid
                            batch_id = str(uuid.uuid4())[:8]
                            
                            batch_data = {
                                "batch_id": batch_id,
                                "batch_name": batch_name,
                                "total": len(cleaned_domains),
                                "completed": 0,
                                "failed": 0,
                                "in_progress": 0,
                                "pending": len(cleaned_domains),
                                "percent": 0,
                                "started_at": datetime.now().isoformat(),
                                "updated_at": datetime.now().isoformat(),
                                "concurrency": concurrency,
                                "domains": cleaned_domains,
                                "stores": {},
                                "status": "pending"
                            }
                            
                            r.set(f"prism:batch:{batch_id}", json.dumps(batch_data), ex=60*60*24*7)  # 7 day TTL
                            
                        # Queue each domain via prism-api
                        api_url = os.getenv("PRISM_API_URL", "https://prism-api-production.up.railway.app")
                        queued = 0
                        errors = []
                        
                        import requests
                        
                        progress_bar = st.progress(0, text="Queuing stores...")
                        
                        for i, domain in enumerate(cleaned_domains):
                            try:
                                url = f"https://{domain}"
                                resp = requests.post(
                                    f"{api_url}/api/v1/jobs",
                                    json={"url": url},
                                    timeout=10
                                )
                                if resp.status_code in (200, 201):
                                    queued += 1
                                else:
                                    errors.append(f"{domain}: HTTP {resp.status_code}")
                            except Exception as e:
                                errors.append(f"{domain}: {str(e)}")
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(cleaned_domains), text=f"Queuing {i+1}/{len(cleaned_domains)}...")
                        
                        # Update batch status
                        batch_data["status"] = "running"
                        batch_data["in_progress"] = queued
                        batch_data["pending"] = len(cleaned_domains) - queued
                        r.set(f"prism:batch:{batch_id}", json.dumps(batch_data), ex=60*60*24*7)
                        
                        progress_bar.empty()
                        
                        if queued > 0:
                            st.success(f"✅ Queued {queued}/{len(cleaned_domains)} stores! Batch ID: `{batch_id}`")
                            st.info("Switch to the **Monitor Batches** tab to track progress.")
                        
                        if errors:
                            with st.expander(f"⚠️ {len(errors)} errors"):
                                for err in errors[:20]:
                                    st.text(err)
                    
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
    
    with tab2:
        st.subheader("Active Batches")
        
        api_url = os.getenv("PRISM_API_URL", "https://prism-api-production.up.railway.app")
        
        # Auto-refresh
        col1, col2 = st.columns([3, 1])
        with col2:
            auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=False, key="batch_refresh")
            if auto_refresh:
                import time
                time.sleep(5)
                st.rerun()
        
        # Find all batch keys
        batch_keys = list(r.scan_iter("prism:batch:*"))
        
        if batch_keys:
            import requests
            
            for key in sorted(batch_keys, reverse=True)[:20]:  # Show latest 20
                try:
                    batch_data = json.loads(r.get(key) or "{}")
                    if not batch_data:
                        continue
                    
                    batch_id = batch_data.get("batch_id", key.replace("prism:batch:", ""))
                    batch_name = batch_data.get("batch_name", "Unknown")
                    domains = batch_data.get("domains", [])
                    total = len(domains) if domains else batch_data.get("total", 0)
                    started_at = batch_data.get("started_at", "")
                    
                    # Query API for actual job status
                    completed = 0
                    failed = 0
                    running = 0
                    pending = total
                    products_total = 0
                    
                    if domains:
                        try:
                            resp = requests.post(
                                f"{api_url}/api/v1/jobs/batch/status",
                                json={"domains": domains[:100]},  # Limit to 100
                                timeout=10
                            )
                            if resp.status_code == 200:
                                status_data = resp.json()
                                completed = status_data.get("completed", 0)
                                running = status_data.get("running", 0)
                                failed = status_data.get("failed", 0)
                                pending = status_data.get("pending", total)
                                products_total = status_data.get("products_total", 0)
                        except Exception as e:
                            st.caption(f"⚠️ Could not fetch status: {e}")
                    
                    # Calculate percent
                    percent = (completed / total * 100) if total > 0 else 0
                    
                    # Determine overall status
                    if completed + failed >= total:
                        overall_status = "completed" if failed == 0 else "completed with errors"
                    elif running > 0:
                        overall_status = "running"
                    else:
                        overall_status = "pending"
                    
                    with st.expander(f"**{batch_name}** ({batch_id}) - {percent:.1f}%", expanded=(overall_status == "running")):
                        # Progress bar
                        st.progress(min(percent / 100, 1.0), text=f"{completed}/{total} stores completed")
                        
                        # Stats row
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Total", total)
                        col2.metric("✅ Done", completed)
                        col3.metric("🔄 Running", running)
                        col4.metric("❌ Failed", failed)
                        col5.metric("📦 Products", products_total)
                        
                        # Status badge
                        status_colors = {"running": "🟢", "completed": "✅", "pending": "🟡", "completed with errors": "🟠"}
                        st.write(f"**Status:** {status_colors.get(overall_status, '⚪')} {overall_status.upper()}")
                        
                        # Timing
                        st.caption(f"Started: {started_at[:19] if started_at else 'N/A'}")
                        
                        # Delete button
                        if st.button(f"🗑️ Delete Batch", key=f"del_{batch_id}"):
                            r.delete(key)
                            st.rerun()
                            
                except Exception as e:
                    st.warning(f"Could not parse batch: {e}")
        else:
            st.info("No active batches. Upload a CSV to start one!")


# Semantic Search Page
elif page == "🔍 Semantic Search":
    st.title("🔍 Semantic Search")
    st.caption("AI-powered product search using vector embeddings")
    
    # API URL configuration
    api_url = os.getenv("PRISM_API_URL", "https://prism-api-production.up.railway.app")
    
    # Search input
    search_query = st.text_input(
        "Search Query",
        placeholder="Try: 'minimalist white watch with red button' or 'laptop for college students'",
        help="Use natural language to describe what you're looking for"
    )
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        search_mode = st.selectbox(
            "Search Mode",
            ["hybrid", "semantic", "text"],
            help="Hybrid combines semantic + filters (recommended)"
        )
    with col2:
        limit = st.number_input("Results", min_value=5, max_value=50, value=10)
    with col3:
        st.write("")  # Spacer
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if search_query and (search_button or st.session_state.get("last_search") != search_query):
        st.session_state["last_search"] = search_query
        
        with st.spinner("Searching..."):
            try:
                import requests
                from urllib.parse import urlencode
                
                # Call prism-api search endpoint
                params = {
                    "q": search_query,
                    "mode": search_mode,
                    "limit": limit
                }
                
                response = requests.get(f"{api_url}/api/v1/search?{urlencode(params)}", timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    hits = data.get("hits", [])
                    
                    if hits:
                        st.success(f"✅ Found {data['total']} results in {data['processing_time_ms']}ms")
                        st.caption(f"Search type: **{data['search_type']}**")
                        
                        # Fetch enriched_data for products
                        product_ids = [h["id"] for h in hits]
                        product_ids_str = ",".join([f"'{pid}'" for pid in product_ids])
                        
                        enriched_df = run_query(f"""
                            SELECT id::text, enriched_data
                            FROM products
                            WHERE id::text IN ({product_ids_str})
                        """)
                        enriched_map = dict(zip(enriched_df["id"], enriched_df["enriched_data"]))
                        
                        # Display results as cards
                        for idx, hit in enumerate(hits):
                            with st.expander(
                                f"**{hit['title']}** - {hit['brand'] or 'Unknown Brand'}", 
                                expanded=(idx == 0)
                            ):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.write(f"**Retailer:** {hit['retailer']}")
                                    if hit.get('price'):
                                        st.write(f"**Price:** ${hit['price']:.2f}")
                                    if hit.get('in_stock') is not None:
                                        stock_emoji = "✅" if hit['in_stock'] else "❌"
                                        st.write(f"**In Stock:** {stock_emoji}")
                                    
                                    st.write(f"**Match Score:** {hit['score']:.3f}")
                                    st.write(f"[🔗 View Product]({hit['url']})")
                                    
                                    # Show semantic summary if available
                                    enriched = enriched_map.get(hit['id'])
                                    if enriched and isinstance(enriched, dict):
                                        summary = enriched.get('semantic_summary')
                                        if summary:
                                            st.info(f"**AI Description:** {summary}")
                                        
                                        # Show key features
                                        features = enriched.get('key_features')
                                        if features:
                                            st.write("**Key Features:**")
                                            for feature in features[:3]:
                                                st.write(f"  • {feature}")
                                
                                with col2:
                                    if hit.get('image_url'):
                                        try:
                                            st.image(hit['image_url'], width=150)
                                        except:
                                            st.caption("Image unavailable")
                                    st.caption(f"ID: `{hit['id'][:8]}...`")
                    else:
                        st.warning("No results found")
                        st.info("💡 Try:\n- Different keywords\n- More general terms\n- Semantic mode for natural language")
                
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.code(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to prism-api")
                st.info(f"Make sure prism-api is running at: **{api_url}**")
                st.code(f"Set PRISM_API_URL environment variable if using different URL")
            except Exception as e:
                st.error(f"Search error: {e}")
                import traceback
                st.code(traceback.format_exc())
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
        st.caption(f"Fields: {', '.join(recent_jobs.columns)}")
        st.dataframe(recent_jobs, use_container_width=True)
    else:
        st.info("No crawl jobs yet")


# Queues Page
elif page == "📬 Queues":
    st.title("📬 Queue Management")
    
    # Check Redis connection
    r = get_redis_client()
    if not r:
        st.error("❌ REDIS_URL not set! Cannot connect to queues.")
        st.info("Set REDIS_URL in your environment variables.")
        st.stop()
    
    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=False)
        if auto_refresh:
            import time
            time.sleep(5)
            st.rerun()
    
    # Queue Stats
    st.subheader("📊 Queue Status")
    
    queue_stats = get_queue_stats()
    total_pending = sum(s.get('pending', 0) for s in queue_stats.values())
    total_failed = sum(s.get('failed', 0) for s in queue_stats.values())
    
    # Queue metrics in columns
    cols = st.columns(6)
    queue_names = ["orchestrate", "platform", "discover", "extract", "index", "price"]
    queue_icons = ["🎯", "🏪", "🔍", "📦", "📇", "💰"]
    
    for i, (queue, icon) in enumerate(zip(queue_names, queue_icons)):
        stats = queue_stats.get(queue, {"pending": 0, "failed": 0})
        pending = stats.get('pending', 0)
        failed = stats.get('failed', 0)
        delta = f"-{failed} failed" if failed > 0 else None
        cols[i].metric(f"{icon} {queue.title()}", pending, delta=delta, delta_color="inverse")
    
    st.divider()
    
    # Active Jobs with Progress
    st.subheader("⏳ Active Jobs")
    
    active_jobs = get_active_jobs()
    if active_jobs:
        for job in active_jobs:
            col1, col2, col3 = st.columns([2, 4, 1])
            with col1:
                st.text(f"🔹 {job['job_id'][:8]}...")
            with col2:
                progress = job['percent'] / 100
                st.progress(progress, text=f"{job['phase']} - {job['percent']:.1f}% ({job['current']}/{job['total']})")
            with col3:
                st.text(job.get('updated_at', '')[:10] if job.get('updated_at') else '')
    else:
        st.info("No active jobs running")
    
    st.divider()
    
    # Dead Letter Queue
    st.subheader("💀 Dead Letter Queue")
    
    dlq_jobs = get_dlq_jobs()
    dlq_count = len(dlq_jobs)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric("Failed Jobs", dlq_count)
    with col2:
        if dlq_count > 0:
            if st.button("🗑️ Clear DLQ", type="secondary"):
                cleared = clear_dlq()
                st.success(f"Cleared {cleared} failed jobs")
                st.rerun()
    
    if dlq_jobs:
        # Show DLQ jobs as expandable
        for job in dlq_jobs[:20]:  # Limit to 20
            with st.expander(f"❌ {job.get('task', 'unknown')} - {job.get('failed_at', '')}"):
                st.json(job)
    
    st.divider()
    
    # Queue Controls
    st.subheader("🎛️ Queue Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Clear Individual Queue**")
        queue_to_clear = st.selectbox("Select queue", queue_names)
        queue_info = queue_stats.get(queue_to_clear, {"pending": 0, "failed": 0})
        queue_count = queue_info.get('pending', 0)
        
        if queue_count > 0:
            if st.button(f"🗑️ Clear {queue_to_clear} ({queue_count} jobs)", type="secondary"):
                cleared = clear_queue(queue_to_clear)
                st.success(f"Cleared {cleared} jobs from {queue_to_clear}")
                st.rerun()
        else:
            st.info(f"{queue_to_clear} queue is empty")
    
    with col2:
        st.write("**Clear All Queues**")
        st.warning(f"⚠️ This will clear {total_pending} pending jobs!")
        
        confirm = st.checkbox("I confirm I want to clear all queues")
        if confirm and total_pending > 0:
            if st.button("☢️ Clear All Queues", type="primary"):
                results = clear_all_queues()
                total_cleared = sum(results.values())
                st.success(f"Cleared {total_cleared} jobs from all queues")
                st.rerun()
    
    st.divider()
    
    # Redis Info
    st.subheader("📈 Redis Info")
    
    redis_info = get_redis_info()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Memory", redis_info.get("memory", "N/A"))
    col2.metric("Clients", redis_info.get("clients", 0))
    col3.metric("Total Keys", redis_info.get("keys", 0))
    col4.metric("Uptime (days)", redis_info.get("uptime", 0))


# Products Page
elif page == "🛍️ Products":
    st.title("🛍️ Products")
    
    # Filters
    col1, col2, col3, col4 = st.columns(4)
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
        has_price = st.checkbox("🏷️ Has Price", value=False)
    with col4:
        limit = st.number_input("Limit", min_value=10, max_value=100000, value=100)
    
    # Build query - ALL fields (excluding removed columns: upc, mpn)
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
            p.gtin,
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
    if has_price:
        query += " AND p.price IS NOT NULL"
    
    query += f" ORDER BY p.first_seen_at DESC LIMIT {limit}"
    
    products = run_query(query)
    
    st.metric("Total Results", len(products))
    
    if not products.empty:
        st.caption(f"Fields ({len(products.columns)}): {', '.join(products.columns)}")
        # Don't use use_container_width=True to enable horizontal scrolling for wide tables
        st.dataframe(products, height=600)
        
        # Download button
        csv = products.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "products.csv", "text/csv")
    else:
        st.info("No products found")


# Store Page
elif page == "🏪 Store":
    # Selection of retailer on top like in products
    st.title("🏪 Store")
    
    # Check if we are in detailed view
    if st.session_state.get("selected_product_id"):
        product_id = st.session_state.selected_product_id
        
        # Back button
        if st.button("⬅️ Back to Store"):
            st.session_state.selected_product_id = None
            st.rerun()
            
        # Fetch product details
        product = run_query(f"""
            SELECT p.*, r.name as retailer_name 
            FROM products p 
            LEFT JOIN retailers r ON p.retailer_id = r.id 
            WHERE p.id = '{product_id}'
        """)
        
        if not product.empty:
            p = product.iloc[0]
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Images
                images = run_query(f"SELECT * FROM product_images WHERE product_id = '{product_id}' ORDER BY position")
                if not images.empty:
                    main_img = images.iloc[0]['stored_url'] or images.iloc[0]['source_url']
                    st.image(main_img, use_container_width=True)
                    
                    # Gallery
                    if len(images) > 1:
                        cols = st.columns(5)
                        for idx, img_row in images.iterrows():
                            with cols[idx % 5]:
                                st.image(img_row['stored_url'] or img_row['source_url'])
                else:
                    st.info("No images for this product")
            
            with col2:
                st.header(p['title'])
                st.subheader(f"{p['brand']} | {p['retailer_name']}")
                
                # Price section
                price = float(p['price']) if p['price'] is not None else None
                orig_price = float(p['original_price']) if p['original_price'] is not None else None
                
                if price is not None:
                    if orig_price and orig_price > price:
                        st.markdown(f"### :red[${price:.2f}] ~~${orig_price:.2f}~~")
                    else:
                        st.markdown(f"### ${price:.2f}")
                else:
                    st.markdown("### Price TBD")
                
                st.write(f"**SKU:** {p['retailer_sku']}")
                st.write(f"**Availability:** {'✅ In Stock' if p['in_stock'] else '❌ Out of Stock'}")
                if p['rating']:
                    st.write(f"**Rating:** {p['rating']} ⭐ ({int(p['review_count'] or 0)} reviews)")
                
                st.divider()
                st.write("**Description:**")
                st.write(p['description'] or "No description available.")
                
                st.link_button("View on Retailer Site", p['url'])
                
            # Price History Chart
            st.divider()
            st.subheader("💰 Price History")
            history = run_query(f"SELECT price, recorded_at FROM product_prices WHERE product_id = '{product_id}' ORDER BY recorded_at")
            if not history.empty:
                history['recorded_at'] = pd.to_datetime(history['recorded_at'])
                st.line_chart(history.set_index('recorded_at')['price'])
            else:
                st.info("No price history recorded yet.")
        else:
            st.error("Product not found.")
            st.session_state.selected_product_id = None
            
    else:
        # Filters at the top
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            search = st.text_input("🔍 Search products", "")
        with col2:
            retailers = run_query("SELECT id, name FROM retailers ORDER BY name")
            if not retailers.empty:
                retailer_options = ["All"] + retailers['name'].tolist()
                selected_retailer = st.selectbox("🏪 Retailer", retailer_options, key="store_retailer")
            else:
                selected_retailer = "All"
        with col3:
            has_price_store = st.checkbox("🏷️ Has Price", value=False, key="store_has_price")
        with col4:
            sort_by = st.selectbox("Sort by", ["Newest", "Price: Low to High", "Price: High to Low"])

        # Build query
        query = """
            SELECT 
                p.id, p.title, p.price, p.original_price, p.brand,
                (SELECT stored_url FROM product_images WHERE product_id = p.id ORDER BY position LIMIT 1) as img_url,
                (SELECT source_url FROM product_images WHERE product_id = p.id ORDER BY position LIMIT 1) as fallback_img
            FROM products p
            LEFT JOIN retailers r ON p.retailer_id = r.id
            WHERE 1=1
        """
        
        if search:
            query += f" AND p.title ILIKE '%{search}%'"
        if selected_retailer != "All":
            query += f" AND r.name = '{selected_retailer}'"
        if has_price_store:
            query += " AND p.price IS NOT NULL"
            
        if sort_by == "Newest":
            query += " ORDER BY p.first_seen_at DESC"
        elif sort_by == "Price: Low to High":
            query += " ORDER BY p.price ASC"
        else:
            query += " ORDER BY p.price DESC"
            
        query += " LIMIT 40"
        
        products = run_query(query)
        
        if not products.empty:
            # Display products in a grid
            cols_per_row = 4
            for i in range(0, len(products), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(products):
                        p = products.iloc[idx]
                        with col:
                            img = p['img_url'] or p['fallback_img']
                            if img:
                                st.image(img, use_container_width=True)
                            else:
                                st.image("https://via.placeholder.com/200x200?text=No+Image", use_container_width=True)
                            
                            st.write(f"**{p['title'][:50]}...**" if len(p['title']) > 50 else f"**{p['title']}**")
                            st.write(f"{p['brand']}")
                            
                            price = float(p['price']) if p['price'] is not None else None
                            orig_price = float(p['original_price']) if p['original_price'] is not None else None
                            
                            if price is not None:
                                if orig_price and orig_price > price:
                                    st.markdown(f"**:red[${price:.2f}]** ~~${orig_price:.2f}~~")
                                else:
                                    st.write(f"${price:.2f}")
                            else:
                                st.write("Price TBD")
                            
                            if st.button("View Details", key=f"btn_{p['id']}"):
                                st.session_state.selected_product_id = p['id']
                                st.rerun()
        else:
            st.info("No products found matching your criteria.")


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
        limit = st.number_input("Limit", min_value=10, max_value=100000, value=100, key="price_limit")
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
        avg_price = prices['price'].mean()
        if pd.notna(avg_price):
            col2.metric("Avg Price", f"${avg_price:.2f}")
        else:
            col2.metric("Avg Price", "N/A")
            
        discounted = prices[prices['original_price'].notna() & (prices['original_price'] > prices['price'])]
        col3.metric("Products with Discounts", len(discounted))
        
        st.caption(f"Fields: {', '.join(prices.columns)}")
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
        st.caption(f"Fields: {', '.join(retailers.columns)}")
        st.dataframe(retailers)
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
        limit = st.number_input("Limit", min_value=10, max_value=100000, value=100, key="url_limit")
    
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
        st.caption(f"Fields: {', '.join(urls.columns)}")
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
        
        st.caption(f"Fields: {', '.join(jobs.columns)}")
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
                try:
                    conn = get_db_connection()
                    with conn.session as session:
                        for table in tables_to_clear:
                            session.execute(f"TRUNCATE TABLE {table} CASCADE")
                        session.commit()
                    st.success(f"✅ Cleared: {', '.join(tables_to_clear)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear tables: {e}")
    
    with col2:
        st.subheader("Clear All Data")
        st.error("🚨 Nuclear option - clears everything!")
        
        confirm_all = st.checkbox("I understand this will delete ALL data")
        if confirm_all and st.button("☢️ Clear All Tables", type="secondary"):
            try:
                conn = get_db_connection()
                with conn.session as session:
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
                            session.execute(f"TRUNCATE TABLE {table} CASCADE")
                        except Exception as e:
                            st.warning(f"Could not clear {table}: {e}")
                    session.commit()
                st.success("✅ All tables cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to clear tables: {e}")


# Footer
st.sidebar.divider()
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
