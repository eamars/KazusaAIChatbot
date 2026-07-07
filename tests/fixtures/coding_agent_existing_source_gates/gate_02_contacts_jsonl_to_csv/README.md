# Contacts JSONL To CSV

Convert JSONL contact records to CSV.

```powershell
python -m contacts_jsonl_to_csv.cli contacts.jsonl contacts.csv
python -m contacts_jsonl_to_csv.cli contacts.jsonl contacts.csv --fields id,name,email
```

When fields are omitted, the first record defines the CSV columns.
