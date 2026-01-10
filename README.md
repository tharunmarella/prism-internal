# 🔮 Prism Dashboard

A Streamlit dashboard to monitor, explore, and manage the Prism product database and worker queues.

## ✨ Features

| Page | Description |
|------|-------------|
| 📊 **Overview** | Product/retailer counts, recent activity, metrics |
| 📤 **Batch Scrape** | Upload CSV of store domains, track bulk scraping progress |
| 🔍 **Semantic Search** | AI-powered product search with vector embeddings |
| 📬 **Queues** | Monitor RabbitMQ queues, DLQ, active jobs with progress |
| 📧 **Lead Outreach** | Manage outreach contacts, send AI-generated preview emails |
| 🧬 **Taxonomy** | Visualize Neo4j category/dimension graph with vis.js |
| 🛍️ **Products** | Search, filter, export products as CSV |
| 🏪 **Store** | Visual product catalog with detail view |
| 🖼️ **Images** | Product image gallery with thumbnails |
| 💰 **Price History** | Track price changes over time |
| 🏪 **Retailers** | Retailer list with product counts |
| 🔗 **Discovered URLs** | Filter by type and extraction status |
| 📋 **Crawl Jobs** | Job history, stats, duration tracking |
| 🗑️ **Clear Data** | Delete specific tables or all data |

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL=postgresql://user:pass@host:5432/prism
export REDIS_URL=redis://localhost:6379/0
export PRISM_API_URL=http://localhost:8000

# Run dashboard
streamlit run app.py

# Or use Docker Compose
docker-compose up
```

### Docker

```bash
# Build
docker build -t prism-dashboard .

# Run
docker run -p 8501:8501 \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  prism-dashboard
```

## 📸 Screenshots

### Overview
- Total products, retailers, URLs, jobs, prices, images
- Recent products table with retailer info
- Recent crawl jobs with status

### Batch Scrape
- Upload CSV with store domains
- Configure concurrency and limits
- Real-time progress tracking
- Monitor active batches

### Semantic Search
- Natural language product search
- Hybrid, semantic, or text modes
- AI-generated descriptions from enrichment
- Key features display

### Queue Management
- Queue depth metrics for all 6 queues
- Active jobs with progress bars
- Dead Letter Queue browser
- Clear individual or all queues
- Redis memory/clients info

### Store View
- Visual product grid with images
- Filter by retailer, price, search
- Product detail page with:
  - Image gallery
  - Price with sale detection
  - Stock status
  - Ratings & reviews
  - Price history chart

## 🔧 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `REDIS_URL` | ⚠️ | Redis URL (needed for queues page) |
| `PRISM_API_URL` | ⚠️ | Prism API URL for queue stats & search |

## 📊 Pages Detail

### 📊 Overview
Quick health check with metrics:
- Products, Retailers, URLs, Jobs, Prices, Images counts
- Recent products table
- Recent crawl jobs

### 📤 Batch Scrape
Upload a CSV file with store domains to queue bulk scraping:

1. **Upload CSV** - Supports columns: `domain`, `current_domain`, `url`
2. **Configure** - Set batch name, concurrency, limit
3. **Monitor** - Track progress, completed, failed counts

### 🔍 Semantic Search
AI-powered product search using Qdrant vectors:

- **Hybrid mode** - Combines semantic + text (recommended)
- **Semantic mode** - Pure vector similarity
- **Text mode** - Traditional full-text search

Displays enriched AI summaries and key features when available.

### 📬 Queues
Full RabbitMQ queue management:

| Queue | Purpose |
|-------|---------|
| orchestrate | Job routing |
| platform | Shopify/WooCommerce API |
| discover | Sitemap crawling |
| extract | Product extraction |
| index | Search indexing |
| price | Price refresh |

Features:
- Queue depth metrics
- Active job progress bars
- Dead Letter Queue browser
- Clear individual/all queues
- Redis server info

### 🛍️ Products
Full product database explorer:

- Filter by retailer, price, search term
- All product fields displayed
- Export to CSV
- Up to 100,000 rows

### 🏪 Store
Visual product catalog:

- Grid view with product images
- Click for full detail page
- Price history chart
- Image gallery
- Retailer links

### 🖼️ Images
Product image gallery:

- Grid of thumbnails
- Shows stored or source URLs
- Product title captions

### 💰 Price History
Price tracking over time:

- Search by product title
- Price and original price
- Stock status at each point
- Export to CSV

### 🏪 Retailers
Retailer management:

- Platform detection (Shopify, WooCommerce, etc.)
- Product counts per retailer
- Crawl settings and last crawled

### 🔗 Discovered URLs
URL discovery tracking:

- Filter by type (product, category, other)
- Filter by extracted status
- Source and depth info

### 📋 Crawl Jobs
Job history and analytics:

- Job type and status
- URLs discovered/crawled
- Products found/updated
- Duration tracking
- Error messages

### 📧 Lead Outreach
Manage outreach to potential merchant customers:

- View/filter leads from `outreach_contacts` table
- Send AI-generated preview emails with enrichment PDFs
- Track status (pending, sent, replied, converted)
- Auto-scheduler sends one email every 5 minutes

### 🧬 Taxonomy
Visualize the Neo4j product taxonomy:

- Interactive graph visualization using vis.js
- Categories (blue nodes) and Dimensions (green nodes)
- `REQUIRES` relationships between them
- Fullscreen mode for detailed exploration

### 🗑️ Clear Data
Database maintenance:

- Clear specific tables
- Clear all tables (with confirmation)
- Row counts before deletion

## 🚢 Deploy to Railway

1. Create new service from this repo
2. Set `DATABASE_URL` environment variable (same as prism-worker)
3. Optionally set `REDIS_URL` for queue features
4. Optionally set `PRISM_API_URL` for search/queue stats
5. Railway auto-detects Dockerfile and deploys
6. Generate domain in Settings → Networking

## 🔗 Related Services

- **[prism-worker](../prism-worker/)** - Background job processor
- **[prism-api](../prism-api/)** - REST API
- **[prism-indexenrich](../prism-indexenrich/)** - Indexing & enrichment
- **[prism-core](../prism-core/)** - Shared library
