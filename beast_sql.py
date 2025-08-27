#!/usr/bin/env python3
# MDTOOLS // SQL OF ALL THINGS - MUTANT BEAST by MDHojayfa
# Author: MDHojayfa | GitHub: MDTOOLS
# For educational and authorized use only.

import os
import platform
import subprocess
import sys
import time
import json
import logging
import argparse
import urllib.parse
import sqlite3
import threading
import signal
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Tuple
from datetime import datetime

HAS_RICH = False
HAS_ALIVE_PROGRESS = False
HAS_OPENAI = False
HAS_PYFIGLET = False
HAS_YASPIN = False
HAS_SKLEARN = False
HAS_NUMPY = False
HAS_SCHEDULE = False

try:
    from rich.console import Console
    from rich.table import Table as RichTable
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.progress import Progress
    HAS_RICH = True
except ImportError:
    pass

try:
    from alive_progress import alive_bar
    from alive_progress.animations.spinners import scrolling
    HAS_ALIVE_PROGRESS = True
except ImportError:
    pass

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    pass

try:
    import pyfiglet
    HAS_PYFIGLET = True
except ImportError:
    pass

try:
    from yaspin import yaspin
    HAS_YASPIN = True
except ImportError:
    pass

try:
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    HAS_SKLEARN = True
    HAS_NUMPY = True
except ImportError:
    pass

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    pass

if HAS_RICH:
    console = Console()
else:
    console = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BANNER_TEXT = "MDTOOLS // SQL OF ALL THINGS"
MOTIVATIONAL_QUOTES = [
    "Legendary things take time. Stay sharp.",
    "In the shadows of code, vulnerabilities hide. Illuminate them ethically.",
    "Defense is the best offense. Scan wisely."
]
CONFIG_FILE = 'sql_beast_config.json'
DEFAULT_CONFIG = {
    "mode": "normal",
    "level": "godlevel",
    "api_keys": {},
    "lab_targets": ["127.0.0.1", "localhost"],
    "repeat": 0,
    "crawl_limit": 0  # 0 for unlimited
}
DEFAULT_TIMEOUT = 10
LAB_ALLOWED = set(DEFAULT_CONFIG["lab_targets"])
VERSION = "beast-1.0"

@dataclass
class Finding:
    id: str
    timestamp: str
    target: str
    endpoint: str
    param: str
    evidence: str
    severity: str
    confidence: float
    tags: List[str]

def typewriter(msg: str, delay: float = 0.01):
    for ch in msg:
        print(ch, end="", flush=True)
        time.sleep(delay)
    print()

def animate_banner():
    if HAS_PYFIGLET:
        print(pyfiglet.figlet_format(BANNER_TEXT))
    else:
        print(BANNER_TEXT)
    typewriter(MOTIVATIONAL_QUOTES[0])

def env_check(show=True):
    is_kali = 'kali' in platform.release().lower()
    is_termux = 'termux' in os.environ.get('PREFIX', '')
    if show:
        lines = [
            f"OS: {platform.system()}",
            f"Python: {platform.python_version()}",
            f"Termux: {'YES' if is_termux else 'NO'}",
            f"Kali: {'YES' if is_kali else 'NO'}",
            f"OpenAI: {'OK' if HAS_OPENAI else 'MISSING'}",
            f"pyfiglet: {'OK' if HAS_PYFIGLET else 'MISSING'}",
            f"yaspin: {'OK' if HAS_YASPIN else 'MISSING'}",
            f"sklearn: {'OK' if HAS_SKLEARN else 'MISSING'}",
            f"numpy: {'OK' if HAS_NUMPY else 'MISSING'}",
            f"rich: {'OK' if HAS_RICH else 'MISSING'}",
            f"alive_progress: {'OK' if HAS_ALIVE_PROGRESS else 'MISSING'}",
            f"schedule: {'OK' if HAS_SCHEDULE else 'MISSING'}"
        ]
        print("\n".join(lines))
    return "full" if is_kali else ("light" if is_termux else "full")

def install_sqlmap():
    if os.path.exists('sqlmap'):
        return os.path.abspath('sqlmap/sqlmap.py')
    try:
        if input("Install sqlmap from GitHub (~10MB)? [y/n]: ").lower() == 'y':
            subprocess.check_call(['git', 'clone', 'https://github.com/sqlmapproject/sqlmap'])
            return os.path.abspath('sqlmap/sqlmap.py')
    except:
        print("Failed to install sqlmap, skipping.")
    return None

def load_plugins(dirpath="plugins"):
    plugins = {}
    if os.path.isdir(dirpath):
        sys.path.insert(0, dirpath)
        for fname in os.listdir(dirpath):
            if fname.endswith(".py"):
                mod_name = fname[:-3]
                try:
                    mod = __import__(mod_name)
                    plugins[mod_name] = mod
                except:
                    logger.error(f"Failed to load plugin {mod_name}")
    return plugins

def auth_confirm(targets, authorized):
    if not authorized:
        for t in targets:
            if input(f"Authorized for {t}? [y/n]: ").lower() != 'y':
                sys.exit("Authorization required.")

def crawl_site(base_url, limit=0, headers=None):
    discovered = set([base_url])
    # Robots
    try:
        resp = requests.get(urllib.parse.urljoin(base_url, "/robots.txt"), headers=headers, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if ':' in line:
                    _, path = line.split(':', 1)
                    discovered.add(urllib.parse.urljoin(base_url, path.strip()))
    except:
        pass
    # Sitemap
    try:
        resp = requests.get(urllib.parse.urljoin(base_url, "/sitemap.xml"), headers=headers, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'xml')
            for loc in soup.find_all('loc'):
                discovered.add(loc.text)
    except:
        pass
    # Recursive links/forms
    visited = set()
    to_visit = [base_url]
    while to_visit:
        url = to_visit.pop(0)
        if url in visited or (limit > 0 and len(discovered) >= limit):
            continue
        visited.add(url)
        try:
            resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
            soup = BeautifulSoup(resp.text, 'html.parser')
            for tag in soup.find_all(['a', 'form']):
                if tag.name == 'a':
                    href = tag.get('href')
                else:
                    href = tag.get('action')
                if href:
                    full = urllib.parse.urljoin(base_url, href)
                    if base_url in full and full not in discovered:
                        discovered.add(full)
                        to_visit.append(full)
        except:
            pass
    return list(discovered)

def baseline_fetch(url, headers):
    try:
        start = time.time()
        resp = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        return resp.text, time.time() - start
    except:
        return "", 0.0

def test_param(url, param, baseline_text, baseline_time, headers):
    variations = ["'", "1' OR '1'='1", "AND SLEEP(5)"]
    for v in variations:
        params = {param: v}
        try:
            start = time.time()
            resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
            dt = time.time() - start
            len_diff = abs(len(resp.text) - len(baseline_text))
            if 'sql' in resp.text.lower() or len_diff > 100 or dt > 5:
                sev = "high" if 'sql' in resp.text.lower() else "medium"
                conf = 0.8 if dt > 5 else 0.6
                tags = ["error-based" if 'sql' in resp.text.lower() else "time-based"]
                return Finding(str(int(time.time() * 1000)), datetime.now().isoformat(), url, url, param, resp.text[:100], sev, conf, tags)
        except:
            pass
    return None

def run_sqlmap(url, sqlmap_path):
    if sqlmap_path:
        cmd = ['python', sqlmap_path, '-u', url, '--batch', '--risk=0', '--level=1', '--smart', '--text-only']
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.stdout
        except:
            return "sqlmap error"
    return ""

class AITriage:
    def __init__(self, findings: List[Finding]):
        self.findings = findings

    def cluster(self):
        if HAS_SKLEARN and HAS_NUMPY:
            if not self.findings:
                return []
            X = np.array([[{"low":0, "medium":1, "high":2}.get(f.severity, 0), f.confidence] for f in self.findings])
            n = min(5, len(self.findings)//2 or 1)
            model = AgglomerativeClustering(n_clusters=n)
            labels = model.fit_predict(X)
            clusters = {}
            for i, lab in enumerate(labels):
                clusters.setdefault(lab, []).append(self.findings[i])
            return list(clusters.values())
        else:
            groups = {}
            for f in self.findings:
                key = f.target + f.endpoint
                groups.setdefault(key, []).append(f)
            return list(groups.values())

    def prioritize(self, clusters):
        ranked = []
        for cl in clusters:
            avg_conf = sum(f.confidence for f in cl) / len(cl) if cl else 0
            sev_w = sum(2 if f.severity == "high" else 1 if f.severity == "medium" else 0 for f in cl) / len(cl) if cl else 0
            score = avg_conf * (1 + sev_w)
            ranked.append((cl, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

def ai_insights(findings, api_client, api_type):
    if not api_client:
        return "AI not available"
    try:
        prompt = f"Summarize findings: {json.dumps([asdict(f) for f in findings])}. Classify, cluster. Ethical: defenses. Unethical awareness: tactics like union-based (combine), error-based (errors), blind (infer), stacked (multi), out-of-band (exfil) - high-level, no instructions."
        response = api_client.chat.completions.create(model="grok-3" if api_type == "grok" else "gpt-4", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except:
        return "AI error"

def ai_suggest(findings, api_client, api_type):
    if not api_client:
        return "AI not available"
    try:
        prompt = f"Suggest ethical SQLi tests for: {json.dumps([asdict(f) for f in findings])}."
        response = api_client.chat.completions.create(model="grok-3" if api_type == "grok" else "gpt-4", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except:
        return "AI error"

class Simulator:
    def __init__(self, lab_targets):
        self.lab_targets = set(lab_targets)

    def allowed(self, host):
        return host in self.lab_targets

    def demo_injection(self):
        conn = sqlite3.connect(':memory:')
        cur = conn.cursor()
        cur.execute("CREATE TABLE users (username TEXT, secret TEXT)")
        cur.execute("INSERT INTO users VALUES ('admin', 'secret')")
        conn.commit()
        query = "SELECT * FROM users WHERE username = 'admin' OR '1'='1'"
        cur.execute(query)
        rows = cur.fetchall()
        conn.close()
        return {"query": query, "rows": rows, "note": "Conceptual demo only"}

    def demo_time(self):
        time.sleep(2)
        return {"type": "time-based", "description": "Simulated delay for blind inference demo", "steps": ["Baseline time", "Payload delay", "Compare"]}

class Reporter:
    def __init__(self, findings, out_md="report.md", out_html="report.html", out_json="report.json", include_advice=False):
        self.findings = findings
        self.out_md = out_md
        self.out_html = out_html
        self.out_json = out_json
        self.include_advice = include_advice

    def save_md(self):
        lines = ["# SQL OF ALL THINGS Report\n", f"Generated: {datetime.now().isoformat()}\n"]
        for f in self.findings:
            lines += [
                f"### {f.id}\n",
                f"- Target: {f.target}\n",
                f"- Endpoint: {f.endpoint}\n",
                f"- Param: {f.param}\n",
                f"- Severity: {f.severity}\n",
                f"- Confidence: {f.confidence}\n",
                f"- Evidence: {f.evidence}\n",
                f"- Tags: {', '.join(f.tags)}\n"
            ]
            lines += ["Ethical: Use prepared statements.\n", "Unethical Awareness: Union-based combine; blind infer; etc. - for defense only.\n"]
            if self.include_advice:
                lines += ["Advice: Input validation, WAF.\n"]
        with open(self.out_md, 'w') as f:
            f.write("\n".join(lines))
        print(f"Saved MD to {self.out_md}")

    def save_html(self):
        lines = ["<html><body><h1>Report</h1>"]
        for f in self.findings:
            lines += [f"<h2>{f.id}</h2><ul><li>Target: {f.target}</li><li>Severity: {f.severity}</li></ul>"]
        lines += ["</body></html>"]
        with open(self.out_html, 'w') as f:
            f.write("\n".join(lines))
        print(f"Saved HTML to {self.out_html}")

    def save_json(self):
        data = {"findings": [asdict(f) for f in self.findings]}
        with open(self.out_json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON to {self.out_json}")

class Runner:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.stop_event = threading.Event()

    def start(self):
        if self.cfg['repeat'] <= 0:
            self.run_once()
            return
        while not self.stop_event.is_set():
            self.run_once()
            for _ in range(self.cfg['repeat']):
                if self.stop_event.is_set():
                    break
                time.sleep(1)

    def run_once(self):
        headers = {"User-Agent": "MDTOOLS"}
        sqlmap_path = install_sqlmap() if env_check(False) == "full" else None
        plugins = load_plugins()
        findings = []
        print("Scanning targets...")
        for i, target in enumerate(self.args.targets):
            print(f"Target {i+1}/{len(self.args.targets)}: {target}")
            endpoints = crawl_site(target, self.cfg['crawl_limit'], headers)
            for j, e in enumerate(endpoints):
                print(f"Endpoint {j+1}/{len(endpoints)}: {e}")
                baseline_text, baseline_time = baseline_fetch(e, headers)
                parsed = urllib.parse.urlparse(e)
                qs = urllib.parse.urlparse_qs(parsed.query)
                for param in qs:
                    f = test_param(e, param, baseline_text, baseline_time, headers)
                    if f:
                        findings.append(f)
                if sqlmap_path:
                    sql_res = run_sqlmap(e, sqlmap_path)
                    if "vulnerable" in sql_res.lower():
                        findings.append(Finding(str(int(time.time() * 1000)), datetime.now().isoformat(), target, e, "", sql_res[:100], "high", 0.9, ["sqlmap"]))
        triage = AITriage(findings)
        clusters = triage.cluster()
        prioritized = triage.prioritize(clusters)
        api_type = "grok" if self.cfg['api_keys'].get("grok") else "openai"
        api_key = self.cfg['api_keys'].get(api_type)
        api_client = OpenAI(base_url="https://api.x.ai/v1" if api_type == "grok" else "https://api.openai.com/v1", api_key=api_key) if api_key and HAS_OPENAI else None
        if api_client:
            insights = ai_insights(findings, api_client, api_type)
            print(insights)
            suggestions = ai_suggest(findings, api_client, api_type)
            print(suggestions)
        sim = Simulator(self.cfg['lab_targets'])
        for cluster, score in prioritized:
            print(f"Cluster Score {score}")
            for f in cluster:
                print(f"{f.id} {f.target} {f.param} {f.severity}")
            if self.args.lab_mode and self.args.authorized:
                if input("Simulate cluster? [y/n]: ").lower() == 'y':
                    for f in cluster:
                        host = urllib.parse.urlparse(f.target).hostname
                        if sim.allowed(host):
                            demo = sim.demo_injection()
                            print(json.dumps(demo, indent=2))
                            time_demo = sim.demo_time()
                            print(json.dumps(time_demo, indent=2))
                        else:
                            print("Not allowed for sim")
        reporter = Reporter(findings, include_advice=self.args.include_advice)
        if input("Generate report? [y/n]: ").lower() == 'y':
            reporter.save_md()
            reporter.save_html()
            reporter.save_json()
        if input("Remediation guide? [y/n]: ").lower() == 'y':
            print("Use prepared statements, WAF. Ref OWASP.")
        typewriter(MOTIVATIONAL_QUOTES[2])

def main():
    parser = argparse.ArgumentParser(description="SQL Beast")
    parser.add_argument("targets", nargs='+', help="Targets")
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--lab-mode", action="store_true")
    parser.add_argument("--include-advice", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    animate_banner()
    env_check()
    auth_confirm(targets, args.authorized)
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_CONFIG
        cfg["api_keys"]["grok"] = input("Grok key: ") or None
        cfg["api_keys"]["openai"] = input("OpenAI key: ") or None
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f)
    if args.self_test:
        args.targets = ["http://testphp.vulnweb.com/"]
    runner = Runner(cfg, args)
    def stop(sig, frame):
        runner.stop_event.set()
    signal.signal(signal.SIGINT, stop)
    runner.start()

if __name__ == "__main__":
    main()
