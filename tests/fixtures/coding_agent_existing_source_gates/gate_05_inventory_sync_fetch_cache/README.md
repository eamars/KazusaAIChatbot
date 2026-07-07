# Inventory Sync

Read an inventory CSV, fetch each vendor page, extract the HTML title and first
`h1`, and write a consolidated CSV report.

```powershell
python -m inventory_sync.cli --input inventory.csv --output report.csv
```

Input columns: `sku`, `name`, `url`.
