from pathlib import Path


def save_report(results, output_dir: Path):
    count = {"green": 0, "yellow": 0, "red": 0}

    for res in results:
        count[res["verdict"]] += 1  

    order = ["red", "yellow", "green"]
    three_worst = sorted(results, key=lambda item: order.index(item["verdict"]))[:3]

    linetext = [
        "Отчет об обработке чертежей",
        f"Всего чертежей: {len(results)}",
        "Из них:",
        f"green = {count["green"]}",
        f"yellow = {count["yellow"]}",
        f"red = {count["red"]}",
        "Три наиболее проблемных чертежа:",
        ]
    
    for num, result in enumerate(three_worst):
        linetext.append(f"№ {num + 1}: {result['drawing']} ({result['verdict']}): {result['reason']}")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(linetext), encoding="utf-8")