# Beast_of_sql
### MDTOOLS
# 🐉 Beast of SQL – Modular SQL Injection Toolkit

**Beast of SQL** is a modular and extensible SQL Injection scanner built for learning, research, and authorized penetration testing.  
It includes crawling, injection testing, reporting, and a **plugin architecture** for extending functionality.

---

## ✨ Features

- 🔍 **Crawling engine** – scans `robots.txt`, `sitemap.xml`, links, and forms.
- 💉 **SQLi detection** – simple payloads + baseline checks (`'`, `OR '1'='1`, time-based).
- 🧠 **AI Triage** – clusters and prioritizes results (supports Grok / OpenAI if API key is set).
- 🎭 **Simulation Mode** – in-memory SQLite demo for injections & time-delays.
- 📊 **Reports** – export findings in **Markdown**, **JSON**, or extend via plugins.
- 🔌 **Plugin support** – add custom scanners, exporters, or integrations without touching core code.
- 🎨 **Banner & UI polish** – `rich`, `pyfiglet`, `alive-progress`, and `yaspin`.

---

## 📂 Project Structure

```
Beast_of_sql/
├── beast_sql.py            # Main script
├── sql_beast_config.json   # Config file
├── requirements.txt        # Dependencies
├── Dependencies.sh         # Shell installer
├── plugins/                # Custom modules
│   ├── extra_scan.py
│   ├── report_exporter.py
│   └── ...
└── README.md
```

---

## ⚡ Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/MDHojayfa/Beast_of_sql.git
   cd Beast_of_sql
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or use the provided installer:

   ```bash
   chmod +x Dependencies.sh
   ./Dependencies.sh
   ```

3. (Optional) Add `sqlmap` for advanced injection testing:

   ```bash
   git clone https://github.com/sqlmapproject/sqlmap
   ```

---

## ⚙️ Configuration

The tool uses `sql_beast_config.json`. Example:

```json
{
  "mode": "normal",
  "level": "godlevel",
  "api_keys": { "grok": null, "openai": null },
  "lab_targets": ["127.0.0.1", "localhost"],
  "repeat": 0,
  "crawl_limit": 0
}
```

- `mode`: normal | lab-mode | self-test  
- `level`: basic → godlevel (affects scanning depth)  
- `api_keys`: API keys for AI analysis (optional)  
- `lab_targets`: test machines or IPs  
- `repeat`: rescan interval (0 = run once)  
- `crawl_limit`: max links to crawl  

---

## 🚀 Usage

- **Self-test mode** (demo only):

  ```bash
  python3 beast_sql.py dummy --authorized --self-test
  ```

- **Scan a local target**:

  ```bash
  python3 beast_sql.py http://127.0.0.1 --authorized --lab-mode
  ```

- **With AI insights** (requires API key):

  ```bash
  python3 beast_sql.py https://target.com --authorized
  ```

---

## 🔌 Plugins

Plugins live in the `plugins/` folder.  
Each plugin should expose a `run()` function.

**Example:**

```python
# plugins/extra_scan.py
def run(target, results):
    print(f"[PLUGIN] Running extra scan on {target}")
    # custom checks here
    return {"extra": "No issues found"}
```

The main script auto-loads and runs all available plugins.  
Use this to add **XSS checks, SSRF scanners, new report formats (CSV, PDF, Slack/Discord exporters), etc.**

---

## 📊 Reports

Built-in exporters:  
- Markdown (`.md`)  
- JSON (`.json`)  

Plugins can extend this to CSV, HTML, PDF, or integrations like **Slack**, **Telegram**, or **Discord**.

---

## ⚠️ Disclaimer

This tool is for **educational purposes and authorized testing only**.  
Do not scan targets without explicit permission.  
The author is not responsible for misuse.

---

## 🛠️ Author

Created by **[MDHojayfa](https://github.com/MDHojayfa)**  
🐉 *"Unleash the beast — responsibly."*
