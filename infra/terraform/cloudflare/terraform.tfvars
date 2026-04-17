region    = "eu-central-1"
zone_name = "shiftbloom.studio"
# Frontend stays at ohi.shiftbloom.studio (this is a CNAME added by dns.tf
# on top of the shiftbloom.studio portfolio site). All OTHER OHI records
# are flattened to `ohi-<label>.shiftbloom.studio` (1-level subdomains,
# covered by free-tier Universal SSL). CF doesn't allow subdomain zones
# on free tier, so we can't just delegate ohi.shiftbloom.studio.
apex_subdomain = "ohi"
# cf_account_id and edge_secret are supplied at apply time via -var
