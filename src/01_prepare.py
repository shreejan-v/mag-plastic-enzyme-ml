"""Build labelled protein set + homology clusters (for leakage-free splits) + DIAMOND all-vs-all.
Positives: PlasticDB (329). Negatives: MAG proteins with NO PlasticDB hit (e<=1e-5).
Hard negatives: negatives whose Prokka product looks like a hydrolase/esterase/lipase/protease.
"""
import re, subprocess, sys
from pathlib import Path
import pandas as pd
from Bio import SeqIO

ROOT = Path.home() / "metagenomics_project"
PDB = ROOT / "mag_reconstruction/publication/data/plasticdb"
FAA = {"bin.3": ROOT / "mag_reconstruction/prokka_output/bin3/bin3.faa",
       "maxbin.002": ROOT / "mag_reconstruction/prokka_output/maxbin002/maxbin002.faa"}
DIAMOND = Path.home() / "miniconda3/envs/prokka_env/bin/diamond"
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
AA = set("ACDEFGHIKLMNPQRSTVWY")
HARD = re.compile(r"hydrolase|esterase|lipase|protease|peptidase|amidase|depolymerase|oxidase|cutinase", re.I)

hit_ids = set(pd.read_csv(PDB / "plasticdb_hits_annotated.tsv", sep="\t")["qseqid"])
rows = []
for rec in SeqIO.parse(PDB / "PlasticDB.fasta", "fasta"):
    f = rec.id.split("|")
    rows.append(dict(id=f"PDB_{f[0]}", seq=str(rec.seq).upper(), label=1, source="PlasticDB",
                     product=f[1], polymer=f[3], hard_neg=False))
for mag, p in FAA.items():
    for rec in SeqIO.parse(p, "fasta"):
        if rec.id in hit_ids:
            continue
        prod = rec.description.split(" ", 1)[1] if " " in rec.description else ""
        rows.append(dict(id=rec.id, seq=str(rec.seq).upper().rstrip("*"), label=0, source=mag,
                         product=prod, polymer="", hard_neg=bool(HARD.search(prod))))
df = pd.DataFrame(rows)
df = df[df.seq.map(lambda s: 50 <= len(s) <= 1500 and set(s) <= AA)].reset_index(drop=True)

# MAG proteins WITH PlasticDB hits (any tier): held out of training, used as external check
ext = []
for mag, p in FAA.items():
    for rec in SeqIO.parse(p, "fasta"):
        if rec.id in hit_ids:
            ext.append(dict(id=rec.id, seq=str(rec.seq).upper().rstrip("*"), source=mag))
ext = pd.DataFrame(ext)
ext = ext[ext.seq.map(lambda s: set(s) <= AA)]
ext.to_csv(OUT / "mag_hit_proteins.csv", index=False)

fa = OUT / "all.faa"
fa.write_text("".join(f">{i}\n{s}\n" for i, s in zip(df.id, df.seq)))
subprocess.run([str(DIAMOND), "makedb", "--in", str(fa), "-d", str(OUT / "all")], check=True, capture_output=True)
subprocess.run([str(DIAMOND), "blastp", "-d", str(OUT / "all.dmnd"), "-q", str(fa), "--very-sensitive",
                "-e", "1e-3", "-k", "200", "--threads", "8", "-o", str(OUT / "allvall.tsv"),
                "--outfmt", "6", "qseqid", "sseqid", "pident", "bitscore", "qcovhsp", "scovhsp"],
               check=True, capture_output=True)
av = pd.read_csv(OUT / "allvall.tsv", sep="\t", names=["q", "s", "pid", "bits", "qcov", "scov"])

# connected components at >=30% id and >=50% coverage both ways -> homology groups
parent = {i: i for i in df.id}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for q, s in av[(av.pid >= 30) & (av.qcov >= 50) & (av.scov >= 50)][["q", "s"]].itertuples(index=False):
    parent[find(q)] = find(s)
df["group"] = df.id.map(find)
df.to_csv(OUT / "dataset.csv", index=False)
print(df.label.value_counts().to_string())
print("hard negatives:", int(df.hard_neg.sum()), "| groups:", df.group.nunique(),
      "| positive groups:", df[df.label == 1].group.nunique(), "| largest group:", df.group.value_counts().iloc[0])
print("external MAG-hit proteins:", len(ext))
