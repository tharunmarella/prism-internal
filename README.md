# Prism Dashboard 🔮

A Streamlit dashboard to monitor and explore the Prism product database.

## Features

- 📊 **Overview** - Product/retailer counts, recent activity
- 🛍️ **Products** - Search, filter, download CSV
- 🏪 **Retailers** - List with product counts
- 🔗 **Discovered URLs** - Filter by type and status
- 📋 **Crawl Jobs** - Job history and stats
- 🗑️ **Clear Data** - Delete specific tables or all data

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set database URL
export DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Run dashboard
streamlit run app.py
```

### Deploy to Railway

1. Create new service from this repo
2. Set `DATABASE_URL` environment variable (same as prism-worker)
3. Railway auto-detects Dockerfile and deploys
4. Generate domain in Settings → Networking

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |

## Screenshots

Access at: `https://your-domain.railway.app`

## Related

- [prism-worker](https://github.com/tharunmarella/prism-worker) - Background job processor
- [prism-api](https://github.com/tharunmarella/prism-api) - REST API

