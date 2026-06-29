# Technical Integration Guide
## TechCorp API v3.0 — Developer Documentation

---

## 1. Authentication

TechCorp API uses **OAuth 2.0** with Bearer tokens for authentication.

### 1.1 Obtaining an API Key
1. Log in to the Developer Portal at developers.techcorp.in
2. Navigate to API Keys → Create New Key
3. Select the required permission scopes
4. API keys are valid for **365 days** and can be rotated manually

### 1.2 Making Authenticated Requests
Include your API key in the Authorization header:
```
Authorization: Bearer YOUR_API_KEY
```

### 1.3 Rate Limits
- Standard tier: **100 requests/minute**, 10,000 requests/day
- Business tier: **500 requests/minute**, 100,000 requests/day
- Enterprise tier: **2,000 requests/minute**, unlimited daily

Exceeding rate limits returns HTTP 429 (Too Many Requests). Retry after the time indicated in the `Retry-After` response header.

---

## 2. Core API Endpoints

### 2.1 User Management
```
GET  /api/v3/users          — List users
POST /api/v3/users          — Create user
GET  /api/v3/users/{id}     — Get user details
PUT  /api/v3/users/{id}     — Update user
DELETE /api/v3/users/{id}   — Deactivate user (soft delete)
```

### 2.2 Data Management
```
POST /api/v3/data/upload     — Upload data file (max 100MB)
GET  /api/v3/data/{id}       — Retrieve data record
GET  /api/v3/data/search     — Search with filters
DELETE /api/v3/data/{id}     — Delete data record
```

---

## 3. Webhooks

TechCorp supports webhooks for real-time event notifications.

### 3.1 Setting Up Webhooks
1. Go to Developer Portal → Webhooks → Add Endpoint
2. Enter your HTTPS endpoint URL
3. Select event types to subscribe to
4. TechCorp signs webhook payloads with HMAC-SHA256 using your webhook secret

### 3.2 Supported Events
- `user.created` — New user registration
- `user.updated` — User profile changes
- `payment.success` — Payment completed
- `payment.failed` — Payment failure
- `data.processed` — Async data processing complete

### 3.3 Retry Policy
Failed webhook deliveries are retried **5 times** with exponential backoff:
- 1st retry: 5 minutes
- 2nd retry: 30 minutes
- 3rd retry: 2 hours
- 4th retry: 8 hours
- 5th retry: 24 hours

---

## 4. Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| 400 | Bad Request | Check request parameters |
| 401 | Unauthorized | Check API key validity |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource does not exist |
| 429 | Rate Limited | Wait and retry |
| 500 | Server Error | Contact support with trace ID |

---

## 5. Data Formats

All API requests and responses use **JSON** (Content-Type: application/json). Dates are in **ISO 8601** format (YYYY-MM-DDTHH:MM:SSZ). Pagination uses cursor-based approach:
- Pass `cursor` parameter for next page
- Maximum `limit` per page: 100 records

---

## 6. SDK Support

Official SDKs are available for:
- **Python**: `pip install techcorp-sdk`
- **JavaScript/Node**: `npm install @techcorp/sdk`
- **Java**: Maven central as `com.techcorp:api-sdk`
- **Go**: `go get github.com/techcorp/sdk-go`

Community SDKs are available for Ruby, PHP, and .NET.

---

## 7. Compliance and Security

- All data is encrypted in transit (TLS 1.3) and at rest (AES-256)
- TechCorp is **SOC 2 Type II** and **ISO 27001** certified
- GDPR and India DPDP Act compliant
- Penetration testing conducted annually by third-party security firms
