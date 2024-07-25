import json
import pandas as pd
import sys

from sklearn.metrics import confusion_matrix, accuracy_score

def main():
    if len(sys.argv) < 2:
        print("Usage: parse_outputs.py [filename.jsonl]")
        return
    
    filename = sys.argv[1]
    content = list()

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            content.append(json.loads(line))

    results = pd.DataFrame(columns=["pair_id", "type", "true", "pred"])
    for i, c in enumerate(content):
        results.loc[i] = [
            c["id"],
            c["type"],
            c["true"],
            c["pred"]
        ]

    print(f"""TOTAL ACCURACY: {accuracy_score(results["true"], results["pred"].map(lambda x: 0 if x not in [1, 2] else x))}""")
    print("BY TYPE:")
    for type in results["type"].unique():
        r = results[results["type"] == type]
        print(f"""{type}: {accuracy_score(r["true"], r["pred"].map(lambda x: 0 if x not in [1, 2] else x))}""")

    print(f"FAILED:")
    print("BY TYPE:")
    for type in results["type"].unique():
        r = results[results["type"] == type]
        print(f"""{type}: {r["pred"].map(lambda c: 1 if c not in [1, 2] else 0).sum()}""")

if __name__ == "__main__":
    main()
