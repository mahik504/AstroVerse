import os
import json
from datetime import datetime

def generate_dashboard(experiments_dir):
    index_path = os.path.join(experiments_dir, "index.json")
    if not os.path.exists(index_path):
        print("No experiments found.")
        return

    with open(index_path, "r") as f:
        experiments = json.load(f)

    # Sort experiments by date descending
    experiments.sort(key=lambda x: x.get("date", ""), reverse=True)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AstroVerse Experiment Dashboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }}
            h1 {{ border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #30363d; }}
            th {{ background-color: #161b22; }}
            tr:hover {{ background-color: #21262d; }}
            .status-completed {{ color: #3fb950; font-weight: bold; }}
            .status-failed {{ color: #f85149; font-weight: bold; }}
            .status-running {{ color: #58a6ff; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>AstroVerse Experiment Dashboard</h1>
        <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <table>
            <thead>
                <tr>
                    <th>Experiment ID</th>
                    <th>Date</th>
                    <th>Dataset</th>
                    <th>Config</th>
                    <th>Epochs</th>
                    <th>Final Loss</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
    """

    for exp in experiments:
        status_class = f"status-{exp.get('status', 'unknown').lower()}"
        loss = exp.get('final_loss')
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        
        html_content += f"""
                <tr>
                    <td>{exp.get('id', 'Unknown')}</td>
                    <td>{exp.get('date', 'Unknown')}</td>
                    <td>{exp.get('dataset', 'Unknown')}</td>
                    <td>{exp.get('config', 'Unknown')}</td>
                    <td>{exp.get('epochs', 'Unknown')}</td>
                    <td>{loss_str}</td>
                    <td class="{status_class}">{exp.get('status', 'Unknown')}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    dashboard_path = os.path.join(experiments_dir, "dashboard.html")
    with open(dashboard_path, "w") as f:
        f.write(html_content)
    
    print(f"Dashboard generated at {dashboard_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=".", help="Experiments directory")
    args = parser.parse_args()
    
    exp_dir = os.path.join(os.path.dirname(__file__), args.dir)
    generate_dashboard(exp_dir)
